import argparse
import io
import math
import os
import pickle
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import TCM
import wandb
import lmdb
from PIL import Image

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        rates = output.get("rates")
        if rates is not None:
            bpp_y = rates["y"]
            bpp_z = rates["z"]
            out["bpp_y"] = bpp_y
            out["bpp_z"] = bpp_z
            out["bpp_loss"] = bpp_y + bpp_z
        else:
            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


class ImageLMDBDataset(Dataset):
    """Dataset reader for LMDB archives containing RGB images."""

    def __init__(self, lmdb_path, transform=None):
        if not os.path.exists(lmdb_path):
            raise RuntimeError(f'Missing LMDB file at "{lmdb_path}"')
        self.lmdb_path = lmdb_path
        self.transform = transform
        self._env = None
        self.keys = self._load_keys()

    def _open_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                subdir=os.path.isdir(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def _load_keys(self):
        env = lmdb.open(
            self.lmdb_path,
            subdir=os.path.isdir(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            raw_keys = txn.get(b"__keys__")
            if raw_keys is None:
                raise RuntimeError(
                    "LMDB is missing '__keys__'. Please rebuild it with key metadata."
                )
            keys = pickle.loads(raw_keys)
        env.close()
        return keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self._open_env()
        key = self.keys[index]
        with self._env.begin(write=False) as txn:
            byteflow = txn.get(key)
        if byteflow is None:
            raise IndexError(f"Failed to load key {key} from LMDB.")
        sample = pickle.loads(byteflow)
        if isinstance(sample, (tuple, list)):
            imgbuf = sample[0]
        elif isinstance(sample, dict):
            imgbuf = sample.get("image") or sample.get("img")
            if imgbuf is None:
                raise ValueError("Dictionary sample missing 'image' key.")
        else:
            imgbuf = sample
        img = Image.open(io.BytesIO(imgbuf)).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = None


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters
    trainable_params = {
        n for n, p in net.named_parameters() if p.requires_grad
    }

    assert len(inter_params) == 0
    assert union_params == trainable_params

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
    type='mse',
    log_interval=200,
    global_step=0,
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d, iteration=global_step)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        log_payload = {
            "train/loss": out_criterion["loss"].item(),
            "train/bpp_loss": out_criterion["bpp_loss"].item(),
            "train/aux_loss": aux_loss.item(),
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        }
        if "bpp_y" in out_criterion:
            log_payload["train/bpp_y"] = out_criterion["bpp_y"].item()
        if "bpp_z" in out_criterion:
            log_payload["train/bpp_z"] = out_criterion["bpp_z"].item()
        if type == 'mse':
            log_payload["train/mse_loss"] = out_criterion["mse_loss"].item()
        else:
            log_payload["train/ms_ssim"] = out_criterion["ms_ssim_loss"].item()
        if "vq" in out_net:
            log_payload["train/vq_perplexity"] = out_net["vq"]["perplexity"].item()
        wandb.log(log_payload, step=global_step)
        global_step += 1

        extra_vq = ""
        if "vq" in out_net:
            extra_vq = f"\tVQ perplexity: {out_net['vq']['perplexity'].item():.2f}"
        if i % log_interval == 0:
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                    f"{extra_vq}"
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                    f"{extra_vq}"
                )

    return global_step


def test_epoch(epoch, test_dataloader, model, criterion, type='mse', global_step=None):
    model.eval()
    device = next(model.parameters()).device
    log_step = global_step if global_step is not None else epoch
    track_vq = getattr(model, "use_vq", False)
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()
        vq_meter = AverageMeter() if track_vq else None

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss().item())
                bpp_loss.update(out_criterion["bpp_loss"].item())
                loss.update(out_criterion["loss"].item())
                mse_loss.update(out_criterion["mse_loss"].item())
                if vq_meter is not None and "vq" in out_net:
                    vq_meter.update(out_net["vq"]["perplexity"].item())

        vq_text = f" |\tVQ perplexity: {vq_meter.avg:.2f}" if vq_meter is not None else ""
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}"
            f"{vq_text}\n"
        )
        payload = {
            "test/loss": loss.avg,
            "test/mse_loss": mse_loss.avg,
            "test/bpp_loss": bpp_loss.avg,
            "test/aux_loss": aux_loss.avg,
            "epoch": epoch,
        }
        if vq_meter is not None:
            payload["test/vq_perplexity"] = vq_meter.avg
        if out_criterion.get("bpp_y") is not None:
            payload["test/bpp_y"] = out_criterion["bpp_y"].item()
        if out_criterion.get("bpp_z") is not None:
            payload["test/bpp_z"] = out_criterion["bpp_z"].item()
        wandb.log(payload, step=log_step)

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()
        vq_meter = AverageMeter() if track_vq else None

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss().item())
                bpp_loss.update(out_criterion["bpp_loss"].item())
                loss.update(out_criterion["loss"].item())
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"].item())
                if vq_meter is not None and "vq" in out_net:
                    vq_meter.update(out_net["vq"]["perplexity"].item())

        vq_text = f" |\tVQ perplexity: {vq_meter.avg:.2f}" if vq_meter is not None else ""
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}"
            f"{vq_text}\n"
        )
        payload = {
            "test/loss": loss.avg,
            "test/ms_ssim": ms_ssim_loss.avg,
            "test/bpp_loss": bpp_loss.avg,
            "test/aux_loss": aux_loss.avg,
            "epoch": epoch,
        }
        if vq_meter is not None:
            payload["test/vq_perplexity"] = vq_meter.avg
        if out_criterion.get("bpp_y") is not None:
            payload["test/bpp_y"] = out_criterion["bpp_y"].item()
        if out_criterion.get("bpp_z") is not None:
            payload["test/bpp_z"] = out_criterion["bpp_z"].item()
        wandb.log(payload, step=log_step)

    return loss.avg


def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 2 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    parser.add_argument(
        "--vq_type",
        type=str,
        default="none",
        choices=["none", "diveq", "sf-diveq"],
        help="Vector quantization strategy to plug into the latent coder",
    )
    parser.add_argument(
        "--vq_codebook_size",
        type=int,
        default=512,
        help="Number of codewords for DiVeQ/SF-DiVeQ",
    )
    parser.add_argument(
        "--vq_sigma2",
        type=float,
        default=1e-3,
        help="Directional noise variance used by DiVeQ/SF-DiVeQ",
    )
    parser.add_argument(
        "--vq_prob_decay",
        type=float,
        default=0.99,
        help="EMA decay used to track codeword usage probabilities",
    )
    parser.add_argument(
        "--vq_warmup_iters",
        type=int,
        default=5000,
        help="Number of initial iterations without quantization (recommended for SF-DiVeQ)",
    )
    parser.add_argument(
        "--vq_init_samples_per_code",
        type=int,
        default=40,
        help="Latent samples per codeword when initializing the codebook",
    )
    parser.add_argument(
        "--vq_replace_threshold",
        type=float,
        default=0.01,
        help="Usage threshold for DiVeQ codebook replacement",
    )
    parser.add_argument(
        "--vq_variant",
        type=str,
        default="original",
        choices=["original", "detach"],
        help="Use the original DiVeQ formulation or the detach variant from Appendix B.2",
    )
    parser.add_argument(
        "--vq_init_cache_batches",
        type=int,
        default=50,
        help="Number of recent batches to cache for codebook initialization (Appendix A.5 recommendation)",
    )
    parser.add_argument(
        "--train_lmdb",
        type=str,
        help="Path to LMDB archive used for the training split (test split stays on ImageFolder)",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    wandb_project = os.environ.get("WANDB_PROJECT", "LIC_TCM")
    wandb_run = wandb.init(
        project=wandb_project,
        config=vars(args),
        name=f"gemini_lambda_{args.lmbda}_N_{args.N}_sfdiveq_512",
    )

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )


    if args.train_lmdb:
        print(f"Loading training samples from LMDB: {args.train_lmdb}")
        train_dataset = ImageLMDBDataset(args.train_lmdb, transform=train_transforms)
    else:
        train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = TCM(
        config=[2,2,2,2,2,2],
        head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0.0,
        N=args.N,
        M=320,
        use_vq=args.vq_type != "none",
        vq_type=args.vq_type,
        vq_codebook_size=args.vq_codebook_size,
        vq_sigma2=args.vq_sigma2,
        vq_prob_decay=args.vq_prob_decay,
        vq_warmup_iters=args.vq_warmup_iters,
        vq_init_samples_per_code=args.vq_init_samples_per_code,
        vq_replace_threshold=args.vq_replace_threshold,
        vq_variant=args.vq_variant,
        vq_init_cache_batches=args.vq_init_cache_batches,
    )
    net = net.to(device)
    wandb.watch(net, log="all", log_freq=100)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    global_step = 0
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        global_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            type,
            global_step=global_step
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, type, global_step=global_step)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])

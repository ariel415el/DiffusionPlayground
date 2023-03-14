import argparse

import torch

from ddpm_v1 import DDPM_Ver1
from ddpm_v2 import DDPM_Ver2
from models.dummy import DummyEpsModel
from models.generic_unet import GenericUnet
from train import train
from utils.common import print_num_params


def get_ddpm_imp(denoiser, args):
    if args.ddpm_imp == "v1":
        ddpm = DDPM_Ver1(denoiser, args.n_steps, args.min_beta, args.max_beta)
    else:
        ddpm = DDPM_Ver2(denoiser, args.n_steps, args.min_beta, args.max_beta)
    return ddpm


def get_denoiser(args):
    if args.denoiser_arch == "Unet":
        denoiser = GenericUnet(scales=(32, 16, 8, 4), c=args.c, n_steps=args.n_steps)
    else:
        denoiser = DummyEpsModel(args.c)

    print("Unet params: ", print_num_params(denoiser))
    return denoiser

def main():
    # Set parameters
    parser = argparse.ArgumentParser(description='Train DDPM')
    # Data parameters
    parser.add_argument('--dataset', default='mnist', help="A path to a folder with subfolders with images. If mnist/cifar the mnist is downloaded and used")
    parser.add_argument('--im_size', type=int, default=32, help="Images are resized to this size")
    parser.add_argument('--c', type=int, default=1, help="Desired number of image channesl 1 turns images to grayscale 3 is RGB")

    # DDPM parameters
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--max_beta', type=float, default=0.02)
    parser.add_argument('--min_beta', type=float, default=10 ** -4)
    parser.add_argument('--denoiser_arch', type=str, default="Unet")
    parser.add_argument('--ddpm_imp', type=str, default='v1')

    # Train parameters
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default="cuda:0")

    # Other
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, default="outputs")

    args = parser.parse_args()
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}\t" + (f"{torch.cuda.get_device_name(0)}"))

    denoiser = get_denoiser(args)

    ddpm = get_ddpm_imp(denoiser, args)

    train(ddpm, args)

if __name__ == '__main__':
    main()
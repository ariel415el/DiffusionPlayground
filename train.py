import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils.data import get_dataset


def train(ddpm, args):
    ddpm.to(args.device)
    dataset = get_dataset(args.data_path, args.im_size, args.c)
    loader = DataLoader(dataset, args.batch_size, shuffle=True)

    optim = Adam(ddpm.parameters(), args.lr)

    step = 0
    train_logger = Logger(args.n_epochs * len(loader), args.out_dir)
    for epoch in range(args.n_epochs):
        for x0, _ in loader:
            x0 = x0.to(args.device)

            loss  = ddpm.compute_loss(x0)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_logger.log(loss.item())

            step += 1
            if step % args.save_freq == 0:
                dump_debug_images(ddpm, args, f"{args.out_dir}/step-{step}.png")
                if train_logger.check_loss():
                    print("Best model (stored)")
                    torch.save(ddpm.state_dict(), "ddpm_model.pt")

def dump_debug_images(ddpm, args, path, batch_normalize_images=False):
    with torch.no_grad():
        ddpm.eval()
        generated_images = ddpm.sample((16, args.c, args.im_size, args.im_size), args.device)
        ddpm.train()

    if batch_normalize_images:
        normalize=True
    else:
        normalize=False
        # per channel normalizaiton
        for i in range(len(generated_images)):
            generated_images[i] -= generated_images[i].min()
            generated_images[i] /= generated_images[i].max()

    save_image(generated_images, path, nrow=4, normalize=normalize)

class Logger:
    def __init__(self, total_setps, outputs_dir):
        self.pbar = tqdm(total=total_setps)
        self.losses = []
        self.best_loss = float("inf")
        self.outputs_dir = outputs_dir
        os.makedirs(outputs_dir, exist_ok=True)

    def log(self, loss):
        self.losses.append(loss)
        self.pbar.update(1)

    def check_loss(self, aggregate_interval=1000):
        step = len(self.losses)

        # Plot loss
        plt.plot(range(step), self.losses)
        plt.savefig(f"{self.outputs_dir}/losses.png")
        plt.clf()

        # Track best loss
        aggrloss = np.mean(self.losses[-aggregate_interval:])
        self.pbar.set_description(f"Aggregated loss at step {step}: {aggrloss:.5f}")
        if aggrloss < self.best_loss:
            self.best_loss = aggrloss
            return True
        return False

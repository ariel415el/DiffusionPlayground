"""Modified from from https://github.com/BrianPulfer/PapersReimplementations/tree/main/ddpm"""
import torch
import torch.nn as nn


class DDPM_Ver1(nn.Module):
    def __init__(self, network, n_steps, min_beta, max_beta):
        super(DDPM_Ver1, self).__init__()
        self.n_steps = n_steps
        self.network = network
        self.register_buffer("betas", torch.linspace(min_beta, max_beta, n_steps))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alpha_bars", torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]))

    def compute_loss(self, x0):
        """Generate a noisy image Xt from  X0 and return the MSE of the estimated noise vs the real one  """
        b=len(x0)

        # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
        eta = torch.randn_like(x0).to(x0.device)
        t = torch.randint(1, self.n_steps, (b,)).to(x0.device)

        # Computing the noisy image based on x0 and the time-step (forward process)
        noisy_imgs = self.run_forward(x0, t, eta)

        # Getting model estimation of noise based on the images and the time-step
        eta_theta = self.run_backward(noisy_imgs, t.reshape(b, -1))

        loss = torch.nn.functional.mse_loss(eta_theta, eta)
        return loss

    def run_forward(self, x0, t, eta=None):
        """ Make input image more noisy (we can directly skip to the desired step) """
        b=len(x0)
        a_bar = self.alpha_bars[t]
        noisy = a_bar.sqrt().reshape(b, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(b, 1, 1, 1) * eta
        return noisy

    def run_backward(self, x, t):
        """ Run each image through the network for each timestep t in the vector t.
        The network returns its estimation of the noise that was added."""
        return self.network(x, t)

    def sample(self, im_shape, device):
        """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
        b = im_shape[0]

        # Starting from random noise
        x = torch.randn(*im_shape).to(device)

        for idx, t in enumerate(list(range(self.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(b, 1) * t).to(device).long()
            eta_theta = self.run_backward(x, time_tensor)

            alpha_t = self.alphas[t]
            alpha_t_bar = self.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn_like(x).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = self.betas[t]
                sigma_t = beta_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

        return x



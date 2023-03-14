import torch
from torch import nn
import torch.nn.functional as F

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding


class ConvBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(ConvBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class Block(nn.Module):
    def __init__(self, d, in_c, out_c):
        super(Block, self).__init__()
        self.conv1 = ConvBlock((in_c, d, d), in_c, out_c)
        self.conv2 = ConvBlock((out_c, d, d), out_c, out_c)
        self.conv3 = ConvBlock((out_c, d, d), out_c, out_c)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class HiddenLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(HiddenLayer, self).__init__()
        self.FC1 = nn.Linear(dim_in, dim_in)
        self.FC2 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.FC1(x)
        x = F.silu(x)
        x = self.FC2(x)
        return x.reshape(len(x), -1, 1, 1)


class GenericUnet(nn.Module):
    def __init__(self, scales=(64, 48, 32, 24, 16), nf=10, c=3, n_steps=1000, time_emb_dim=100):
        super(GenericUnet, self).__init__()
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.scales = scales
        self.n = len(self.scales)
        nfs = [c] + [nf * i for i in range(1, self.n + 1)]
        # First half
        down_layers = dict()
        for i in range(self.n):
            down_layers[f"conv{i}"] = Block(self.scales[i], nfs[i], nfs[i+1])
            down_layers[f"te{i}"] = HiddenLayer(time_emb_dim, 1)


        up_layers = dict()
        for i in range(1, self.n):
            up_layers[f"conv{i}"] = Block(self.scales[-i - 1], nfs[-i] + nfs[-i - 1], nfs[-i -1])
            up_layers[f"te{i}"] = HiddenLayer(time_emb_dim, 1)

        self.down_layers = nn.ModuleDict(down_layers)
        self.up_layers = nn.ModuleDict(up_layers)
        self.conv_out = nn.Conv2d(nfs[1], c, 3, 1, 1)


    def forward(self, x, t):
        t = self.time_embed(t)
        xs = []
        for i in range(self.n):
            x_down = F.interpolate(x, size=self.scales[i])
            t_emb = self.down_layers[f'te{i}'](t)
            x =  self.down_layers[f'conv{i}'](x_down + t_emb )
            xs.append(x)

        # print(x.shape)

        for i in range(1, self.n):
            x_up = F.interpolate(x, size=self.scales[-i - 1])
            x = torch.cat([xs[-i - 1], x_up], dim=1)
            t_emb = self.up_layers[f'te{i}'](t)
            x = self.up_layers[f'conv{i}'](x + t_emb)

        out = self.conv_out(x)
        return out


if __name__ == '__main__':
    x = torch.ones(5, 3, 64, 64)
    t = torch.ones(5, dtype=torch.long)
    d = GenericUnet(scales=(64,48,32,24,16), nf=10, c=3)
    # b = MyBlock((3, 64,64), 3, 6)

    print(d(x, t).shape)
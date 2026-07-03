import torch

from torch import nn
from torch.nn import functional as F
from torch.nn import utils
import math

def timestep_embedding(timesteps, dim, max_period=18):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    timesteps = timesteps/max_period
    max_period=100
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# def timestep_embedding(timesteps, dim, max_period=150):
#     """
#     Create sinusoidal timestep embeddings.

#     :param timesteps: a 1-D Tensor of N indices, one per batch element.
#                       These may be fractional.
#     :param dim: the dimension of the output.
#     :param max_period: controls the minimum frequency of the embeddings.
#     :return: an [N x dim] Tensor of positional embeddings.
#     """
#     embedding = timesteps/max_period
      
#     freqs =   (0 * torch.arange(start=0, end=dim, dtype=torch.float32) +1).to(device=timesteps.device)
#     args = timesteps[:, None].float() * freqs[None]
#     embedding = args
#     return embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_embed_dim, n_class=None, downsample=False):
        super().__init__()

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(
                time_embed_dim,
                out_channel,
            ),
        )

        self.conv1 = nn.Conv2d(
                in_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            )


        self.conv2 = nn.Conv2d(
                out_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            )


        self.class_embed = None

        if n_class is not None:
            class_embed = nn.Embedding(n_class, out_channel * 2 * 2)
            class_embed.weight.data[:, : out_channel * 2] = 1
            class_embed.weight.data[:, out_channel * 2 :] = 0

            self.class_embed = class_embed

        self.skip = None

        if in_channel != out_channel or downsample:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=False)
            )

        self.downsample = downsample

        self.silu = SiLU()

    def forward(self, input, emb=None, class_id=None):
        out = input

        out = self.conv1(out)
        emb_out = self.emb_layers(emb).type(out.dtype)
        while len(emb_out.shape) < len(out.shape):
            emb_out = emb_out[..., None]

        if self.class_embed is not None:
            embed = self.class_embed(class_id).view(input.shape[0], -1, 1, 1)
            weight1, weight2, bias1, bias2 = embed.chunk(4, 1)
            out = weight1 * out + bias1

        out = out+emb_out  
        out = self.silu(out)

        out = self.conv2(out)

        if self.class_embed is not None:
            out = weight2 * out + bias2

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        out = self.silu(out)

        return out


class IGEBM(nn.Module):
    def __init__(self, mid_channel= 128, n_class=None):
        super().__init__()

        self.mid_channel = mid_channel

        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)

        

        time_embed_dim = mid_channel * 4
        self.time_embed = nn.Sequential(
            nn.Linear(mid_channel, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )


        self.blocks = nn.ModuleList(
            [
                ResBlock(mid_channel, mid_channel, time_embed_dim, n_class, downsample=True),
                ResBlock(mid_channel, mid_channel, time_embed_dim, n_class),
                ResBlock(mid_channel, mid_channel*2, time_embed_dim, n_class, downsample=True),
                ResBlock(mid_channel*2, mid_channel*2, time_embed_dim, n_class),
                ResBlock(mid_channel*2, mid_channel*2, time_embed_dim, n_class, downsample=True),
                ResBlock(mid_channel*2, mid_channel*2, time_embed_dim, n_class),
            ]
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(
                time_embed_dim,
                mid_channel*2,
            ),
        )
        self.silu = SiLU()
        self.linear = nn.Linear(mid_channel*2, 1)
        

    def forward(self, input, time=None, class_id=None):
        out = self.conv1(input)

        out = self.silu(out)
        emb = self.time_embed(timestep_embedding(time, self.mid_channel))

        for block in self.blocks:
            out = block(out, emb, class_id)

        emb_out = self.emb_layers(emb).type(out.dtype)
        while len(emb_out.shape) < len(out.shape):
            emb_out = emb_out[..., None]

        out = out + emb_out
        out = self.silu(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out


def get_ebm_model():
    return IGEBM()
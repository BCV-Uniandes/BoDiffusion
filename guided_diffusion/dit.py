from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import time
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization12,
    timestep_embedding,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: list,  # to accept generic data types
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim[0] * spacial_dim[1] + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim, bias=True)

    def forward(self, x):
        # bs, njoints, nfeats, nframes = x.shape
        bs, nfeats, nframes, njoints = x.shape
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        x = x.permute((0, 2, 3, 1)).reshape(bs, nframes, njoints*nfeats)
        
        # x = self.poseEmbedding(x)  # [seqlen, bs, d]
        x = self.poseEmbedding(x)  # [bs, seqlen, njoints, nfeats]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats, out_channels):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        self.featsFinal = nn.Linear(self.nfeats, out_channels, bias=True)

    def forward(self, output):
        bs, nframes, d = output.shape
        
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(bs, nframes, self.njoints, self.nfeats)  # nframes, bs, self.njoints, self.nfeats
        
        output = self.featsFinal(output)
        output = output.permute(0, 3, 1, 2)  # [bs, nfeats, nframes, njoints]
        return output


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ValidProjection(nn.Module):
    """
    Module to perform a linear transformation per dimension to ensure
    it will be divisible by (2 ** max(channel_mult)).
    E.g., if dims=2 (so an image), I in R^{26x52} and max(channel_mult)=4
    then ValidProjection(I) in R^{32x64}.
    :param dims: convolutional dimensions (1D, 2D or 3D)
    :param max_mult: max(channel_mult)
    :param shape: input shape
    :param inv: inverse operation. To transform from R^{32x64} to R^{26x52}.
                given the previous example.
    """
    def __init__(self, dims, max_mult, shape, inv):
        super().__init__()
        self.dims = dims
        self.max_mult = max_mult
        self.shape = shape
        if not inv:
            in_out = [(shape[i], math.ceil(shape[i] / (2 ** max_mult)) * (2 ** max_mult)) for i in range(len(shape))]
        else:
            in_out = [(math.ceil(shape[i] / (2 ** max_mult)) * (2 ** max_mult), shape[i]) for i in range(len(shape))]

        self.projections = nn.Sequential(*[nn.Linear(*s, bias=False) for s in in_out])

        self.dims_njoints = in_out[0]
        # self.dims_njoints = in_out[1]

    def forward(self, x):
        
        # total_dims = len(x.shape) - len(self.shape)
        # permute = list(range(len(self.shape) + total_dims))
        # for i in range(1, len(self.shape) + total_dims):
        #     # permutation to set the last dimension the operation we want
        #     # e.g. if projecting dim 3 (0, 1, 2, 3, 4) -> (0, 1, 2, 4, 3)
        #     unchanged_dims = permute[:i]
        #     dim = i
        #     changed_dims = permute[(i + 1):]
            
        #     change = (*(unchanged_dims + changed_dims), dim)

        x = self.projections(x)

            # # permutation to restore the original config
            # # e.g. (0, 1, 2, 4, 3) -> (0, 1, 2, 3, 4)

            # depermute = (*(unchanged_dims + [-1] + [d - 1 for d in changed_dims]),)
            # x = x.permute(depermute)

        return x
    

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    breakpoint()
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _broadcast_to_h(emb, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    emb = emb.squeeze().view(256,1,1,-1)
    bs, _, njoints, nfeats = broadcast_shape
    return emb.expand((bs, 1, njoints, nfeats))


class TransformerModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """
    def __init__(self, njoints, nfeats, model_channels=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", legacy=False, data_rep='rot', dataset='amass', 
                 arch='trans_enc', emb_trans_dec=False, image_size=128,
                 channel_mult=(1, 2, 4, 8), dims=2, in_dim=[41, 3], use_fp16=False, mlp_ratio=4.0, pos_enc=False,
                 init_weights=False, **kargs):
        super().__init__()

        self.legacy = legacy
        
        self.njoints = njoints
        self.nfeats = nfeats
        
        self.data_rep = data_rep
        self.dataset = dataset

        self.latent_dim = model_channels
        print(f'the latent_dim is {model_channels}')
        self.ff_size = ff_size
        print(f'the number of layers is {num_layers}')
        print(f'the number of heads is {num_heads}')
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats =  kargs.get('in_channels', None)
        self.output_feats =  kargs.get('out_channels', None)

        self.normalize_output = kargs.get('normalize_encoder_output', False)
        
        self.pos_enc = pos_enc
        self.init_weights = init_weights

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.batch =  kargs.get('batch_size', None)

        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        
        # self.in_projection = ValidProjection(dims, max(channel_mult), image_size, False)
        # self.out_projection = ValidProjection(dims, max(channel_mult), image_size, True)
        
        # self.reshape_njoints = self.in_projection.dims_njoints[-1]
        
        self.in_latdims = self.njoints * self.input_feats
        self.input_process = InputProcess(self.in_latdims, self.latent_dim)
       
        self.pos_embed = nn.Parameter(th.zeros(1, self.batch, self.latent_dim), requires_grad=False)
        # self.pos_embed = PositionalEncoding(self.latent_dim)
        self.emb_trans_dec = emb_trans_dec

        self.blocks = nn.ModuleList([
            DiTBlock(self.latent_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(self.num_layers)
        ])

        self.out = OutputProcess(self.in_latdims, self.latent_dim, self.njoints, self.input_feats, self.output_feats)
        self.embed_timestep = TimestepEmbedder(self.latent_dim)

        self.image_size = image_size
        self.channel_mult = channel_mult
        self.dtype = th.float16 if use_fp16 else th.float32
        self.in_dim = in_dim

        # self.output = nn.Sequential(
        #     normalization12(self.input_feats),
        #     nn.SiLU(),
        #     zero_module(conv_nd(dims, self.input_feats, self.output_feats, 3, padding=1)),
        # )
        
        if self.init_weights:
            self.initialize_weights()

    def pos_enc_initialize(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.batch ** 0.5))
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(1))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)  

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.embed_timestep.mlp[0].weight, std=0.02)
        nn.init.normal_(self.embed_timestep.mlp[2].weight, std=0.02)
        print('weights init')

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)
        self.input_process.apply(convert_module_to_f16)
        self.out.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        pass


class JointEmbedder(nn.Module):
    def __init__(self, inner_dim, output_dim, cond_dim=6):
        super().__init__()
        self.gru = nn.GRU(input_size=3 * inner_dim,
                          hidden_size=output_dim,
                          batch_first=True)
        self.joint_projection = nn.Conv1d(cond_dim, inner_dim, kernel_size=1, bias=False)
        self.inner_dim = inner_dim

    def forward(self, x):
        # x with shape [b, feat, time, joint]
        B, F, T, J = x.shape
        x = x.reshape(B, F, -1)
        x = self.joint_projection(x)
        x = x.reshape(B, self.inner_dim, T, J).permute((0, 2, 3, 1))  # [bs, T, CJ (3C)]
        x = self.gru(x.reshape(B, T, -1))[1].squeeze(0)  # must be shape [b, output_dim]
        return x


class TransformerJoints(TransformerModel):
    def __init__(self, joint_cond=False, cond_dim=6, **kwargs):
        super().__init__(**kwargs)
        if joint_cond:
            print('**************** we have joint_cond')
            self.window_emb = JointEmbedder(self.latent_dim, self.latent_dim, cond_dim)
        self.joint_cond = joint_cond
        self.joint_embedding = None

    def precompute_joint_embedding(self, joints):
        self.joint_embedding = self.window_emb(joints)

    def reset_joint_embedding(self):
        self.joint_embedding = None

    def forward(self, x, timesteps, joints=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N x C x W x 3] Tensor of additional joints.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        if self.joint_cond:
            if self.joint_embedding is None:
                emb = emb + self.window_emb(joints)
            else:
                emb = emb + self.joint_embedding
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        x = self.in_projection(x)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        h = self.out_projection(h)
        h = self.out(h)

        return h


class ConditionProjection(nn.Module):
    """
    Module to perform a linear transformation per dimension to ensure
    it will be divisible by (2 ** max(channel_mult)).
    E.g., if dims=2 (so an image), I in R^{26x52} and max(channel_mult)=4
    then ValidProjection(I) in R^{32x64}.
    :param dims: convolutional dimensions (1D, 2D or 3D)
    :param max_mult: max(channel_mult)
    :param shape: input shape
    :param inv: inverse operation. To transform from R^{32x64} to R^{26x52}.
                given the previous example.
    """
    def __init__(self, in_dims, max_mult, shape, inv):
        super().__init__()
        self.in_dims = in_dims[0]
        self.max_mult = max_mult
        self.shape = shape[0]
        # if not inv:
        #     in_out = [(in_dims[i], math.ceil(shape[i] / (2 ** max_mult)) * (2 ** max_mult)) for i in range(len(shape))]
        # else:
        #     in_out = [(math.ceil(shape[i] / (2 ** max_mult)) * (2 ** max_mult), in_dims[i]) for i in range(len(shape))] 

        self.projections = nn.Linear(self.in_dims, self.shape, bias=False)

    def forward(self, x):
        
        # total_dims = len(x.shape) - len(self.shape)
        # permute = list(range(len(self.shape) + total_dims))
        # for i in range(1, len(self.shape) + total_dims):
        #     # permutation to set the last dimension the operation we want
        #     # e.g. if projecting dim 3 (0, 1, 2, 3, 4) -> (0, 1, 2, 4, 3)
        #     unchanged_dims = permute[:i]
        #     dim = i
        #     changed_dims = permute[(i + 1):]
            
        #     change = (*(unchanged_dims + changed_dims), dim)

        x = self.projections(x)

            # # permutation to restore the original config
            # # e.g. (0, 1, 2, 4, 3) -> (0, 1, 2, 3, 4)

            # depermute = (*(unchanged_dims + [-1] + [d - 1 for d in changed_dims]),)
            # x = x.permute(depermute)

        return x

class TransformerCondition(TransformerJoints):
    def __init__(self, add_cond=False, **kwargs):
        super().__init__(**kwargs)

        if add_cond:
            print('------------------- we have add_cond')
            self.cond_emb = ConditionProjection(self.in_dim, max(self.channel_mult), self.image_size, False) 
        self.add_cond = add_cond
        self.condition_embedding = None
        self.timer = 0
        
        # self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

    def precompute_joint_embedding(self, joints):
        if self.joint_cond:
            self.joint_embedding = self.window_emb(joints)
        if self.add_cond:
            self.condition_embedding = self.cond_emb(joints)
    
    def forward(self, x, timesteps, joints=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N x C x W x 3] Tensor of additional joints.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # timer = time.time()
        # hs = []
        # adding the timestep embed
        emb = self.embed_timestep(timesteps)                   # (N, D)

        if self.joint_cond:
            if self.joint_embedding is None:
                emb = emb + self.window_emb(joints)
            else:
                emb = emb + self.joint_embedding
        # x = self.in_projection(x)
        
        if self.add_cond:
            if self.condition_embedding is None:
                x = th.cat((x, self.cond_emb(joints)), 1)
            else:
                x = th.cat((x, self.condition_embedding), 1)
        h = x.type(self.dtype)
        h = self.input_process(h)
        x = h + self.pos_embed
        
        for block in self.blocks:
            x = block(x, emb)   
        
        # h = self.final_layer(x, emb)                # (N, T, patch_size ** 2 * out_channels)
        h = self.out(x)
        # h = self.out_projection(h)
        # h = self.output(h)
        # self.timer += time.time() - timer

        return h

import torch
import torch.nn.functional as F
from torch import nn, einsum
from collections import OrderedDict
from einops import rearrange, repeat
from transformers import PreTrainedModel
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from transformers.utils import ModelOutput
from .configuration_tf4ctr import TF4CTRConfig

# helpers
ACT2CLS = {
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)
        
ACT2FN = ClassInstantier(ACT2CLS)

def exists(val):
    return val is not None

def get_activation(activation):
    if activation in ACT2FN:
        return ACT2FN[activation]
    else:
        raise KeyError(f"function {activation} not found in ACT2FN mapping {list(ACT2FN.keys())}")

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out), attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout)),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = x + attn_out
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)
# mlp

class MLP(nn.Module):
    def __init__(self, dims, act = "relu"):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            cur_act = get_activation(act)
            layers.append(cur_act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# main class

class TabTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_shared_categ_embed = True,
        shared_categ_dim_divisor = 8.   # in paper, they reserve dimension / 8 for category shared embedding
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)

        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)

        # take care of shared category embed

        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std = 0.02)

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.norm = nn.LayerNorm(num_continuous)

        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous

        hidden_dimensions = [input_size * t for t in  mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act = mlp_act)

    def forward(self, x_categ, x_cont, return_attn = False, return_last_hidden = False):
        xs = []

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            categ_embed = self.category_embed(x_categ)

            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b = categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim = -1)

            x, attns = self.transformer(categ_embed, return_attn = True)

            flat_categ = rearrange(x, 'b ... -> b (...)')
            xs.append(flat_categ)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if self.num_continuous > 0:
            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        x = torch.cat(xs, dim = -1)
        logits = self.mlp(x)

        rt = (logits,)
        if return_attn:
            rt = rt + (attns,)
        if return_last_hidden:
            rt = rt + (x,)
        return rt


@dataclass
class TF4CTROutput(ModelOutput):
    """
    Output type of [`TF4CTRModel`].

    Args:
        logits: `torch.FloatTensor` of shape `(batch_size, 1)`.
        loss: *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(batch_size, 1)`.
        last_hidden_states:  *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`, 
            `torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`.
        attentions: *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`, 
            `torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`, 
    """
    logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    last_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[torch.FloatTensor] = None


class TF4CTRModel(PreTrainedModel):
    config_class = TF4CTRConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = TabTransformer(
            categories = config.categories,
            num_continuous = config.num_continuous,
            dim = config.hidden_dim,
            dim_out = config.dim_out,
            depth = config.n_layers,
            heads = config.n_heads,
            attn_dropout = config.attention_dropout,
            ff_dropout = config.dropout,
            mlp_hidden_mults = config.mlp_hidden_mults,
            mlp_act = config.activation,
        )

    def forward(self, 
                cat_tensor, 
                dense_tensor, 
                labels=None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
        ):
        # print(f"cat_tensor: {cat_tensor}")
        # print(f"dense_tensor: {dense_tensor}")
        # print(f"labels: {labels}")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(cat_tensor, dense_tensor, return_attn=output_attentions, return_last_hidden=output_hidden_states)
        logits = outputs[0]

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        if not return_dict:
            return ((loss,) + outputs) if loss is not None else outputs
        
        return TF4CTROutput(
            logits=logits,
            loss=loss,
            attentions=outputs[1] if output_attentions else None,
            last_hidden_states=outputs[2] if output_hidden_states else None,
        )
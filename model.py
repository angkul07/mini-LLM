import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, 
                 num_heads, qkv_bias: False, use_flash_attention: False):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.use_flash_attention = use_flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        
        self.dropout_rate = dropout
        self.attn_dropout = nn.Dropout(dropout)

        if not self.use_flash_attention:
            self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_keys(x)
        values = self.W_values(x)

        # Reshape: (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.use_flash_attention:
            context_vec = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values,
                attn_mask=None,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=True
            )
        else:
            attn_scores = queries @ keys.transpose(-2, -1)
            attn_scores = attn_scores / math.sqrt(self.head_dim)
            
            
            mask_slice = self.mask[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_slice, float("-inf"))
            
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            context_vec = attn_weights @ values

        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift 

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(cfg["drop_rate"])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            use_flash_attention=cfg.get("flash", False)
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_norm = self.norm1(x)
        attn_output = self.att(x_norm)
        x = shortcut + self.drop_shortcut(attn_output)

        shortcut = x
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = shortcut + self.drop_shortcut(ff_output)
        return x

class GPTModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying
        # self.tok_emb.weight = self.out_head.weight


    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape

        if seq_len > self.cfg["context_length"]:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds model's context length "
                f"({self.cfg['context_length']})"
            )

        tok_embeds = self.tok_emb(in_idx)
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_indices)
        
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
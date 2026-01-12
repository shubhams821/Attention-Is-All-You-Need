import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads 
        self.head_dim = dim // num_heads
        
        #combined QKV
        self.W_QKV = nn.Linear(dim, dim*3)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        qkv = self.W_QKV(x)
        q, k, v = qkv.chunk(3, dim = -1)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        att = q @ k.transpose(-2,-1)/(self.head_dim**0.5)
        if mask is not None:
            att = att.masked_fill(mask==0, float("-inf"))
        att = F.softmax(att, dim =-1)
        
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(out)



class MultiHeadAttentionDec(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads 
        self.head_dim = dim // num_heads
        
        #combined QKV
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x_e, x):
        assert x_e.shape[0] == x.shape[0]
        assert x_e.shape[-1] == x.shape[-1]

        B, T1, C = x.shape
        _, T2, _ = x_e.shape
        
        q = self.W_Q(x)
        k = self.W_K(x_e)
        v = self.W_V(x_e)
        
        q = q.view(B, T1, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T2, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T2, self.num_heads, self.head_dim).transpose(1,2)

        att = q @ k.transpose(-2,-1)/(self.head_dim**0.5)
        att = F.softmax(att, dim =-1)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B, T1, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
      return self.net(x)



class Encoder(nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.masked_attn = MultiHeadAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.dec_attn = MultiHeadAttentionDec(dim, heads)
        self.ln3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_dim)
    
    def forward(self, x_e, x, mask):
        x = x + self.masked_attn(self.ln1(x), mask)
        x = x + self.dec_attn(x_e, self.ln2(x))
        x = x + self.ff(self.ln3(x))
        return x


class AttIsAllYouNeed(nn.Module):
    def __init__(self, vocab_size, dim = 256, heads = 4, depth = 8, max_len = 256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        
        self.EncBlocks = nn.ModuleList(
            Encoder(dim, heads, hidden_dim = 4*dim)
            for i in range(depth)
        )
        self.DecBlocks = nn.ModuleList(
            Decoder(dim, heads, hidden_dim = 4*dim)
            for i in range(depth)
        )
        self.ln_final = nn.LayerNorm(dim)
        self.heads = nn.Linear(dim, vocab_size)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, idx_in, idx_out):
        B, T = idx_in.shape
        tok_emb = self.token_emb(idx_in)
        pos_emb = self.pos_emb(torch.arange(T, device = idx_in.device))
        
        x_e = tok_emb + pos_emb
        for block in self.EncBlocks:
            x_e = block(x_e)
        
        B, T = idx_out.shape
        tok_emb = self.token_emb(idx_out)
        pos_emb = self.pos_emb(torch.arange(T, device = idx_out.device))

        x = tok_emb + pos_emb
        mask = self.mask[:,:,:T, :T]
        for block in self.DecBlocks:
            x = block(x_e, x, mask)
        
        out = self.ln_final(x)
        logits = self.heads(x)
        return logits

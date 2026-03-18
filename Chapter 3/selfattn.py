import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
    )

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # x: (seq_len, d_in)

        keys    = x @ self.W_key      # (seq_len, d_out)
        queries = x @ self.W_query    # (seq_len, d_out)
        values  = x @ self.W_value    # (seq_len, d_out)

        # Attention scores
        attn_scores = queries @ keys.T   # (seq_len, seq_len)

        # Scaled dot-product attention
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5),
            dim=-1
        )  # (seq_len, seq_len)

        # Context vectors
        context_vec = attn_weights @ values  # (seq_len, d_out)

        return context_vec

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # x: (seq_len, d_in)

        keys    = self.W_key(x)      # (seq_len, d_out)
        queries = self.W_query(x)    # (seq_len, d_out)
        values  = self.W_value(x)    # (seq_len, d_out)

        # Attention scores
        attn_scores = queries @ keys.T   # (seq_len, seq_len)

        # Scaled dot-product attention
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5),
            dim=-1
        )  # (seq_len, seq_len)

        # Context vectors
        context_vec = attn_weights @ values  # (seq_len, d_out)

        return context_vec
    

d_in = inputs.shape[1] #B
d_out = 2

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# queries = sa_v2.W_query(inputs) #A
# keys = sa_v2.W_key(inputs)
# attn_scores = queries @ keys.T
# attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)

# context_length = attn_scores.shape[0]
# mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
# masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

# attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
# print(attn_weights)

# dropout = torch.nn.Dropout(0.5) #A
# print(dropout(attn_weights))

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads                                          #A
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)                                     #B
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)                                                        #C
        queries = self.W_query(x)                                                   #C
        values = self.W_value(x)                                                    #C

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)              #D
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)          #D
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)        #D

        keys = keys.transpose(1, 2)                                                 #E
        queries = queries.transpose(1, 2)                                           #E
        values = values.transpose(1, 2)                                             #E

        attn_scores = queries @ keys.transpose(2, 3)                                #F
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]                      #G
        attn_scores.masked_fill_(mask_bool, -torch.inf)                             #H

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)                       #I
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)      #J
        context_vec = self.out_proj(context_vec)                                    #K
        return context_vec

batch = torch.stack((inputs, inputs), dim=0)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
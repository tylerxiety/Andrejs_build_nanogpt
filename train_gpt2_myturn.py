from dataclasses import dataclass
import torch.nn as nn
import math
import torch
from torch.nn import functional as F
import time

# TDODO
# watch/read attention code in previous video again



class CausalSelfAttention(nn.Module):
    # multi-headed attention
    # optimized for efficiency in pytorch
    # same naming convention to hugging face
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # **Added by chatgpt: Autoregressive mask to prevent attention to future positions**
        self.register_buffer("bias", torch.tril(torch.ones((config.block_size, config.block_size), dtype=torch.uint8)).view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        # attension ( materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
       
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # autoregressive mask
        att = F.softmax(att, dim=-1) # normalize

        y = att @ v # weighted sum:(B, nh, T, T) X (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side. perform the concatenat ion function.
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    # nn.GELU(approximate='tanh') is an approximate version of GELU that was developed due to
    # the error function erf was very slow in tensorflow back to years ago.
    # nowadays we prefer use the exact version and there is no big difference.
    # but back then GPT2 and BERT picked up the approximate version.
    # we are reproducing GPT2 so using the approximate version here.

    # why use GELU instead of RELU: because of the dead relu neuron problem 
    # where any activations that fall on the flat tail at zero will get 
    # zero gradient. But the GELU always contributes a local gradient.
    # GELU empirically work better in practice as demonstrated in the GELU,
    # GPT2 and BERT papers.

    # more modern networks like LLAMA further use SwiGLU, etc.
    # 
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        # at init, scale the weights of residual layers by a factor of 1/sqrt(num of residual layers).
        # because the variance of the activations in the residual stream grows, by sqrt(num of residual layers) if random
        # set a flag here.
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # in layernorm, use the default init in pytorch, scale to be 1 and offset to be 0
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    # OG transformer: LN is after attn or FF, and LN is inside the residual stream
    # GPT2: clean residual stream from supervision all the way down to the inputs, 
    # which is desirable, because the gradients from top flow straight to the inputs unchanged.
    # The attn is an aggregation func, weighted sum, like reduce
    # whereas mlp happens at every single token individually, like map,
    # so the transformer is like a repeteated application of map reduce.
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) # aka FFN
        return x
    
@dataclass
class GPTConfig:
    # block_size: int = 256 # max sequence length
    # vocab_size: int = 65 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    # n_layer: int = 6 # number of layers
    # n_head: int = 6 # number of heads
    # n_embd: int = 384 # embedding dimension

    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # create module transformer, the same to the transformer container in huggingface,
        # using nn.ModuleDict: so that we can index submodule using keys.
        # nn.Embedding is just a fancy wraper around tensor.
        # nn.ModuleList: so that we can index it using integers, same to the schema in huggingface. 12 layers (h.0 - h.11) in the huggingface transformer.
        # ln_f: GPT2 has a final LN at the end, which is different to the OG transformer paper.
        # The other main difference is that LN is moved to before the MLP in GPT2.
        # lm_head: GPT2 uses no bias in the final projection.
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing scheme
        # why tie the token embedding at the bottom of transformer and lm head at the top of the transformer togather:
        # you want these 2 matrices behave similar in the following sense. Similar tokens should be nearly in the token embedding space, similarly, they also should have the same/similar probablity at the output of a transformer.
        # Another reason is, this saves 30% parameters, so it's more efficient in terms of training.
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        # call .apply() to iterate the submodule and apply _init_weights()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # use 0.02, the same to the GPT2 code
            # typically, if following the xavier init, std = 1/sqrt(768 model dimensions) = 0.036. So 0.02 is roughly in the same vicinity.
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 2 times here, because every layer has 2 blocks that add to the residual pathway: attn and mlp.
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                # in pytorch the default init is uniform
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



# -----------------------------------------------------------------------------
import tiktoken
# import numpy as np

# def load_tokens(filename):
#     npt = np.load(filename)
#     npt = npt.astype(np.int32) # added after video
#     ptt = torch.tensor(npt, dtype=torch.long)
#     return ptt

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and stoe them in memory
        # default on CPU
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"load {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T 
        # if loading the next batch would be out of bounds, rest
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


# -----------------------------------------------------------------------------

# attempt to autodetetct the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
# for apple silicon
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'using device: {device}')

 
# for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# train
# track time
t_s = time.time()

train_loader = DataLoaderLite(B=4, T=32)

# enc = tiktoken.get_encoding('gpt2')
# with open('input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1]) # +1 to include the label for the last token.
# buf = buf.to(device) # can't directly do .to() like model, because it's a tensor and not stateful.
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

# get logits
model = GPT(GPTConfig())
model.to(device)


# optimize!
# AdamW is a bug-fixed version of Adam. 
# keeps 2 buffers: m, v momentum, like RMSprop.
# faster than SGD
# overfit a single batch: 5 rounds: mps: 1.435 loss, 7.20 seconds. cpu: 1.470 loss, 17.1533 seconds; 50 rounds : mps: 0.0028 loss, 23.20 seconds. cpu: 0.0028 loss, 148.49 seconds
# iterate batches: 50 rounds : mps: 6.5409 loss, 23.644 seconds. cpu: 6.5205 loss, 152.89seconds.
# at this scale, most of the loss gain comes from deleting the usage of tokens that never occur (by driving the biases of all the logits that never occur to -inf.)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device) # move tensors to GPU/MPS
    optimizer.zero_grad() # always to zero the grads first
    logits, loss = model(x, y)
    loss.backward() # deposit grads(i.e., do += on grads), accumulates the grads from this loss
    optimizer.step() # to update the param
    print(f"step {i}, loss: {loss.item()}")


# will get loss: 10.9403
# sanity check: the init loss should be around -ln(1/50257) = 10.8, 
# as uniform probability at initialization.
# cpu: 2.1077, mps: 5.656, interesting
# logits, loss = model(x, y)
# print(loss) 

t_e = time.time()
t_dff = t_e - t_s

print(f'time: {t_dff}')
import sys; sys.exit(0)

# # prefix tokens
# num_return_sequences = 5
# max_length = 30

# model= GPT.from_pretrained('gpt2')
# model.eval()
# model.to(device)

# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(5, 1) # (5, 8)
# x = tokens.to(device)

# # generate!
# torch.manual_seed(42)
# torch.mps.manual_seed(42)
# # torch.cuda.manual_seed(42)
# t_s = time.time()
# while x.size(1) < max_length: # max_length=30
#     # forward the model to get the logits
#     with torch.no_grad():
#         logits = model(x) # (B, T, vocab_size)
#         # take the logits at the last position
#         logits = logits[:, -1, :] # (B, vocab_size)
#         # get the probabilities
#         probs = F.softmax(logits, dim=-1)
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select a token from the top-k probabilities
#         # note: multinomial does not demand the input to sum to 1
#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim=1)

# t_e = time.time()
# t_dff = t_e - t_s

# print(f'time: {t_dff}')
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)




# check loading huggingface GPT2 param
# model= GPT.from_pretrained('gpt2')
# print("didn't crash yay!")
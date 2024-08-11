from dataclasses import dataclass
import torch.nn as nn
import math
import torch
from torch.nn import functional as F
import inspect

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
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention, equivalent to the following 4 lines of code but much faster. Not supported on MPS.
        # # attension ( materializes the large (T, T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # autoregressive mask
        # att = F.softmax(att, dim=-1) # normalize
        # y = att @ v # weighted sum:(B, nh, T, T) X (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side. perform the concatenat ion function.
        # output projection
        y = self.c_proj(y)
        return y

# # not using it, for demo purpose only
# class TanhGELU(nn.Module):
#     def forward(self, input):
#         return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


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
    # so d_head = n_embd / n_head = 768 / 12 = 64
    # using GPT3 Small params

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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Start with all of the candidiate parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim group. Any params that is 2D will be decayed, otherwise no decay.
        # i.e. all weight tensors in matmuls + embeddings decay, all bias, layernorm params, scales don't decay.
        # Use weight decay as a regularization.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': decay_params, 'weight_decay': weight_decay}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed param tensors: {len(decay_params)}, with {num_decay_params:,} params")
        print(f"num non-decayed param tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")
        # Create AdamW optimizer and use the fused version if available.
        # Fused version is a lot faster, but not available on all platforms.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


# -----------------------------------------------------------------------------
import tiktoken
# import numpy as np

# def load_tokens(filename):
#     npt = np.load(filename)
#     npt = npt.astype(np.int32) # added after video
#     ptt = torch.tensor(npt, dtype=torch.long)
#     return ptt

class DataLoaderLite:
    # sample data without replacement within the epoch.
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
import time
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
# MPS backend out of memory when using (B=16, T=1024) on macbook air
# mps: step 1, loss: 9.525256156921387, dt: 5741.08ms, tok/sec: 713.45
# cpu: step 1, loss: 9.525256156921387, dt: 103424.57ms, tok/sec: 39.60
# use "nice" number: round up the vocab_size 50257 to nearest "nice" number that has a lot of power of 2, eg 50304. Got 4% speedup in the video. But got worse on MPS, may due to the memory.

# The following speedup techniques are not supported on the MPS:
# 1. lower precisions: precision options for training: FP32, TF32, BF16, FP16; INT8 is for inference.
# recommend read: Automatic Mixed Precision pytorch doc
# torch.autocast(device_type=device, dtype=torch.bfloat16), mps not supported
# using bfloat16 (in stead of float16) is minimual in code change since no need for gradient scaler
# under the hood, some matrix multuply like operations are converted to BF16, but a lot of operations remain in float32(e.g, layernorm, softmax, etc)
# 2. use torch.compile(model)
# 3. flash attention
# 4. fused AdamW optimizer

# gradient accumulation
total_batch_size = 524288 # 2**19, nice number, ~0.5 million tokens
# B = 16 # micro batch size
B = 4 # micro batch size on the MPS
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "make sure the total batch size is divisible by B * T"
grad_acc_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size} ")
print(f"gradient accumulation steps: {grad_acc_steps}")

train_loader = DataLoaderLite(B=B, T=T) # GPT2 max seq length

torch.set_float32_matmul_precision('high') # TF32 available on macbook air with M1 according to doc, but the tok/sec is almost unchanged.

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
# model = GPT(GPTConfig(vocab_size=50304)) # rounded up the vocab_size 50257 to nearest "nice" number that has a lot of power of 2. Got 4% speedup in the video.
model = GPT(GPTConfig()) 
model.to(device)
# model = torch.compile(model) # mps not supported for complie. 3X faster in the video.
# The speedup comes from: 1. reduce python overhead.The python interpreter run layer by layer (i.e., ego mode). The torch.compile compiles the entire neural net as a single object without python interpreter involved, and runs the code efficiently. 2. optimize the read/write between GPU and HBM/global memory (reducing the number of intermediate memory writes and reads).

# Learning rate schedule: cosine decay with linear warmup, as in the GPT3 paper.
max_lr = 6e-4 # use the lr for GPT3 small
min_lr = max_lr * 0.1
# copilot wrote this, where does this come from: warmup_tokens = 375e6 # 375 million tokens for GPT-2, "0.5" in the video is a typo.
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # not exactly the same as the GPT3 paper, but close enough 
    # 1) linear warmup for warmup_steps steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps # +1 to avoid 0 learning rate
    # 2) if it > max_steps, return min_lr
    if it >= max_steps:
        return min_lr
    # 3) in between, do cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)




# optimize!
# AdamW is a bug-fixed version of Adam. 
# keeps 2 buffers: m, v momentum, like RMSprop.
# faster than SGD
# overfit a single batch: 5 rounds: mps: 1.435 loss, 7.20 seconds. cpu: 1.470 loss, 17.1533 seconds; 50 rounds : mps: 0.0028 loss, 23.20 seconds. cpu: 0.0028 loss, 148.49 seconds
# iterate batches: 50 rounds : mps: 6.5409 loss, 23.644 seconds. cpu: 6.5205 loss, 152.89seconds.
# at this scale, most of the loss gain comes from deleting the usage of tokens that never occur (by driving the biases of all the logits that never occur to -inf.)
# Use GPT3 hyper params, due to no details from the GPT2 paper.
# GPT3 and GPT2 are very similar in terms of the architecture, 
# expect the doubled context length, some hyper params change, and was trained a lot longer on a bigger dataset.
# Learning rate: cosine decay with linear warmup.
# Batch size schedule: skipped. Becasue it complicates the arithmetic, not a major improvement, 
# more of a systems and speed improvemnet not a algorithmic optimization improvement.
# Use weight decay of 0.1, as in the GPT3 paper.
# Note that the relationship among the hyper params, weight decay, learning rate, batch size,Adam params beta1, beta2, epsilon, etc. is very complicated.
# Gradient accumulation: we want to use the batch size of 0.5 million as per the GPT3 paper, but the GPU won't fit. So we use a smaller batch size and accumulate the gradients over multiple steps.
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

for step in range(50):
    t0 = time.time()
    optimizer.zero_grad() # always to zero the grads first
    loss_accum = 0.0
    for micro_step in range(grad_acc_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device) # move tensors to GPU/MPS
        # bfloat16 is not sypprted on MPS
        # with torch.autocast(device_type=device, dtype=torch.bfloat16)
        #     logits, loss = model(x, y)
        logits, loss = model(x, y)
        # import code; code.interact(local=locals()) # interactively check
        loss = loss / grad_acc_steps # because the loss is averaged over the batch, due to the reduction='mean' in the cross_entropy function.
        loss_accum += loss.detach() # use detach to avoid the grad computation.
        loss.backward() # deposit grads(i.e., do += on grads), accumulates the grads from this loss
        # Clip the grads to prevent exploding grads. 
        # Global norm of all the grads: square root of the sum of the squares of the grads.
        # Why clip the grads: sometimes the grads can be very large unluckily, and the optimizer can't handle it. It's just a hacky solution.
        # Norm can be high in the beginning then decreases and stabilizes at the value around 1,
        # because the model is random and mostly leaning the biases of the output tokens.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

    # determine and set the learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step() # to update the param
    torch.mps.synchronize() # waiting for the device to finish
    t1 = time.time()
    dt = (t1 - t0) * 1000 # in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T * grad_acc_steps) / dt
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


# will get loss: 10.9403
# sanity check: the init loss should be around -ln(1/50257) = 10.8, 
# as uniform probability at initialization.
# cpu: 2.1077, mps: 5.656, interesting
# logits, loss = model(x, y)
# print(loss) 

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
## 0. Exploring the GPT-2 (124M) OpenAI checkpoint

wte: 
- weights for token embedding. 50257 vocab, 768 dimensions. 
- lookup table for tokens
- h.0-11, 12 layers

wpe: 
- weights for position embedding. 
- max sequence length 1024, up to 1024 positions each token can be attending to in the past. 
- every one of the positions has a fixed vector of 768, which is learned by optimization
- a lookup table for the positions

## 1. Reproduce GPT2
GPT2 architecture:

*Layer Norm*: 
- LN before attn and MLP(FFN), so is outside the residual stream. An additional LN after the final attn and right before the final classifer. While in OG transformer, LN is after attn and MLP, so is inside the residual stream.
- Clean residual stream from supervision all the way down to the inputs, which is desirable, because the gradients from top flow straight to the inputs unchanged.

*GELU*:
- nn.GELU(approximate='tanh') is an approximate version of GELU that was developed due to the error function erf was very slow in tensorflow back to years ago. nowadays we prefer use the exact version and there is no big difference. But back then GPT2 and BERT picked up the approximate version. we are reproducing GPT2 so using the approximate version here.
- why use GELU instead of RELU: because of the dead relu neuron problem 
where any activations that fall on the flat tail at zero will get 
zero gradient. But the GELU always contributes a local gradient.
GELU empirically work better in practice as demonstrated in the GELU,
GPT2 and BERT papers.
- more modern networks like LLAMA further use SwiGLU, etc.

*Tokenizer*:
- gpt2 tokenizer has a compression ratio of roughly 3:1.

*AdamW*:
- a 'bug-fixed' version of Adam.

*Weight sharing scheme*:
- self.transformer.wte.weight = self.lm_head.weight
- why tie the token embedding at the bottom of transformer and lm head at the top of the transformer togather: you want these 2 matrices behave similarly in the following sense. Similar tokens should be nearly in the token embedding space, similarly, they also should have the same/similar probablity at the output of a transformer.
- Another reason is, this saves 30% parameters, so it's more efficient in terms of training.

*Techniques*:
- load data: 
    - always fetch B*T+1 to get the label for the last token.
    - load data to CPU first, ship to GPU when needed to save GPU memory.
- always to zero the grads first.
- sanity check the loss to ensure random initialization of the model is correct.
- move model and data to the same device, note the difference:
    - model.to(device)
    - buf = torch.tensor(tokens[:B*T + 1]) buf = buf.to(device) # can't directly do .to() like model, because it's a tensor and not stateful.


*Insights from Andrej*:
- The attn is an aggregation func, weighted sum, like reduce, whereas mlp happens at every single token individually, like map, so the transformer is like a repeteated application of map reduce.
- 
    
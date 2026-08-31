import json
import math
import os
import torch 
import torch.nn as nn
import torch.distributed as dist # torch.distributed is used for distributed training (example for multi-GPU training)
from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_hidden_layers: int = 24        # total transformer hidden layers in the model
    num_experts: int = 32              # number of experts in the MoE (Mixture of Experts)
    expert_per_token: int = 4          # how many experts each token is routed to
    vocab_size: int = 201088           # size of the token vocabulary
    hidden_size: int = 2880            # dimension of hidden representations
    intermediate_size: int = 2880      # size of feedforward intermediate layer
    swiglu_limit: float = 7.0          # clipping limit for SwiGLU activation
    head_dim: int = 64                 # dimension per attention head
    num_attention_heads: int = 64      # total number of attention heads
    num_key_value_heads: int = 8       # number of heads used for key/value projections
    sliding_window: int = 128          # local attention window size
    initial_context_length: int = 4096 # max sequence length at initialization
    rope_theta: float = 150000.0       # base frequency for RoPE embeddings
    rope_scaling_factor: float = 32.0  # scaling factor for RoPE extension
    rope_ntk_alpha: float = 1.0        # NTK scaling alpha for RoPE
    rope_ntk_beta: float = 32.0        # NTK scaling beta for RoPE


class RMSNorm(nn.Module):
    """ 
    RMSNorm is a normalization technique used in transformer models to normalize the activations of a layer.
    It computes the root mean square of the input tensor and scales it by a learnable parameter.
    RMSNorm(x) = (x / RMS(x)) * γ,  where RMS(x) = sqrt(mean(x²) + ε), where epsilon is a small constant to avoid division by zero, and γ is a learnable scaling parameter.
    """
    def __init__(self, num_features : int, eps : float = 1e-05, device : torch.device | None = None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features, f"Expected input with last dimension {self.num_features}, but got {x.shape[-1]}"
        t,dtype = x.float(), x.dtype # convert to float for numerical stability
        t = t * torch.rsqrt(torch.mean(t*t, dim=-1, keepdim=True) + self.eps) # compute RMS and normalize
        return (t* self.scale).to(dtype)


def _apply_rotary_emb( 
    x : torch.Tensor, cos : torch.Tensor , sin : torch.Tensor,
) - > torch.Tensor :
    """
    Applies rotary positional embeddings to the input tensor x using the provided cosine and sine tensors.
    Rotary embeddings are a way to encode positional information in transformer models by rotating the input representations in a complex plane.

    Args:
        x (torch.Tensor): The input tensor of shape [batch_size, seq_len, num_heads, head_dim].
        cos (torch.Tensor): The cosine tensor of shape [seq_len, head_dim].
        sin (torch.Tensor): The sine tensor of shape [seq_len, head_dim].
    returns:
        torch.Tensor: The input tensor x with rotary embeddings applied, of the same shape as x
    """
    cos = cos.unsqueeze(-2).to(x.dtype) # unsqueeze to match the shape of x for broadcasting , -2 is the second last dimension (num_heads)
    sin = sin.unsqueeze(-2).to(x.dtype) # unsqueeze to match the shape of x for broadcasting
    x1,x2 = torch.chunk(x,2,dim=-1)
    o1 = x1 * cos - x2 * sin # o1  is to  apply the rotation to the first half of the input tensor
    o2 = x1 * sin + x2 * cos # o2 is to apply the rotation to the second half of the input tensor
    return torch.cat([o1,o2],dim=-1) # concatenate the rotated halves

class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding is a module that generates rotary positional embeddings for transformer models.
    RotaryEmbedding works by creating a set of sinusoidal embeddings based on the input sequence length and the specified embedding dimension.
    
    """
    def __init__(self, head_dim : int,
        base : int, dtype : torch.dtype,
        initial_context_length : int=4096,
        scaling_factor : float =1.0,
        ntk_alpha:float = 1.0,
        ntk_beta:float = 32.0,device = torch.device | None=None):
        """
        head_dim - the dimension of each attention head
        base - the base frequency for the rotary embeddings
        dtype - the data type 
        initial_context_length - the maximum sequence length for which to precompute the embeddings
        scaling_factor - a scaling factor for the embeddings 
        ntk_alpha - the alpha parameter for the NTK scaling of the embeddings
        ntk_beta - the beta parameter for the NTK scaling of the embeddings
        device - the device on which to create the embeddings

        """
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.register_buffer("cos_cached", torch.zeros(initial_context_length, head_dim, dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", torch.zeros(initial_context_length, head_dim, dtype=dtype), persistent=False)
        self._update_cos_sin_cache(initial_context_length)

    def _compute_concentration_and_inv_freq(self) ->  torch.Tensor:
        """ 
        Computes the concentration and inverse frequency for the rotary embeddings based on the scaling factor and NTK parameters.
        The concentration is computed as 0.1 * log(scaling_factor) + 1.0, which allows for a more flexible representation of positional information in the embeddings.
        The inverse frequency is computed as base ** (torch.arange(0, head_dim, 2) / head_dim), which generates a set of frequencies that are evenly spaced in the complex plane.
        why it is computed : The concentration and inverse frequency are computed to enable the rotary embeddings to capture positional information in a more flexible and effective manner, especially when dealing with long sequences or varying context lengths.
        YaRN paper: https://arxiv.org/abs/2309.00071 
        why are we computing the concentration and inverse frequency: The concentration and inverse frequency are computed to enable the rotary embeddings to capture positional information in a more flexible and effective manner, especially when dealing with long sequences or varying context lengths. 
        The concentration allows for scaling of the frequencies based on the scaling factor, while the inverse frequency generates a set of frequencies that are evenly spaced in the complex plane, which is essential for the rotary embedding mechanism.
        """
        freq = self.base ** (
            torch.arange(0,self.head_dim, 2, dtype = torch.float, device = self.device) / self.head_dim
        ) # freq - > compute the base frequencies for the rotary embeddings, with a step of 2 to account for the sine and cosine pairs such that the frequencies are evenly spaced in the complex plane

        if self.scaling_factor > 1.0 : 
            concentration = ( 
                0.1 * math.log(self.scaling_factor) + 1.0
            ) # YaRN concentration means that the frequencies are scaled by a factor of 0.1 * log(scaling_factor) + 1.0, which allows for a more flexible representation of positional information in the embeddings

            d_half = self.head_dim / 2 # d_half is half the dimension of the head, used to scale the frequencies for the rotary embeddings .
            
            #ntk by parts NTK means that the frequencies are scaled by a factor of (ntk_alpha * log(ntk_beta) + 1.0) / d_half, which allows for a more flexible representation of positional information in the embeddings
            # NTK stands for Neural Tangent Kernel, which is a theoretical framework for understanding the behavior of neural networks in the infinite-width limit. The NTK scaling allows for better generalization and stability in training deep transformer models.
            low = ( 
                d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base) # d_half
            )
            high = (d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2*math.pi)) / math.log(self.base) )
            assert 0 < low < high < d_half -1, f"Invalid NTK scaling parameters: low={low}, high={high}, d_half={d_half}, initial_context_length={self.initial_context_length}, ntk_alpha={self.ntk_alpha}, ntk_beta={self.ntk_beta}, base={self.base}"

            interpolation = 1.0 / (self.scaling_factor * freq) # interpolation is used to scale the frequencies based on the scaling factor and the base frequencies, allowing for a more flexible representation of positional information in the embeddings.
            extrapolation = 1.0 / freq # extrapolation is meant to scale the frequencies based on the base frequencies, allowing for a more flexible representation of positional information in the embeddings.

            ramp = (
                torch.arange(d_half,dtype = torch.float32,device = freq.device) - low
            ) / (high - low) #ramp means that the frequencies are scaled based on their position in the head dimension, allowing for a more flexible representation of positional information in the embeddings.
            mask  = 1 - ramp.clam(0,1) 
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else : 
            concentration = 1.0 
            inv_freq = 1.0 / freq 
        return concentration, inv_freq 


    def _compute_cos_sin(self,num_tokens : int) :
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device) / concentration
        freqs = torch.einsum("i,j->ij", t, inv_freq) # compute the outer product of t and inv_freq to get the frequencies for each token and head dimension
        cos = freqs.cos() * concentration # freqs.cos -> compute the cosine of the frequencies and scale by the concentration
        sin = freqs.sin() * concentration # freqs.sin -> compute the sine of the frequencies and scale by the concentration
        return cos, sin

    def forward(
            self, query : torch.Tensor ,key : torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
            num_tokens = query.shape[0]
            cos,sin = self._compute_cos_sin(num_tokens)
            query_shape = query.shape 
            query = query.view(num_tokens,-1,self.head_dim) # reshape query to [num_tokens, num_heads, head_dim]
            query  = _apply_rotary_emb(query,cos,sin)
            query = query.reshape(query_shape)

            key_shape = key.shape 
            key = key.view(num_tokens,-1,self.head_dim) # reshape key to [num_tokens, num_heads, head_dim]
            key  = _apply_rotary_emb(key,cos,sin)
            key = key.reshape(key_shape)
            return query,key

def sqpa(Q, K, V, S, sm_scale, sliding_window=0):
    """
    Compute scaled dot-product attention with optional sliding window.

    Parameters
    ----------
    Q : torch.Tensor
        Query tensor of shape (num_tokens, num_heads, head_dim).
    K : torch.Tensor
        Key tensor of shape (num_tokens, num_heads, head_dim).
    V : torch.Tensor
        Value tensor of shape (num_tokens, num_heads, head_dim).
    S : torch.Tensor
        Optional mask or score modifier.
    sm_scale : float
        Scaling factor applied to the dot-product scores.
    sliding_window : int, default=0
        Size of the local window for attention. 
        - If 0, each token attends to all other tokens (full attention).
        - If >0, each token only attends to a fixed neighborhood of nearby tokens,
          reducing computation and focusing on local context.

    Returns
    -------
    torch.Tensor
        Output tensor after applying scaled dot-product attention.

    Notes
    -----
    Sliding window attention is useful for long sequences:
    - It reduces memory and compute cost by limiting attention scope.
    - It emphasizes local context, which is often more relevant in tasks
      like language modeling, audio, or time-series data.
    - It makes attention scale more efficiently to very long inputs.
    """
    # sliding_window == 0 means no sliding window (full attention)

    # Extract dimensions from Q: 
    # n_tokens = sequence length, n_heads = number of heads,
    # q_mult = query multiplier (extra queries per head), d_head = head dimension
    n_tokens, n_heads, q_mult, d_head = Q.shape

    # Ensure K and V have the expected shapes (no q_mult dimension yet)
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)

    # Add a new axis for q_mult and expand K and V so they match Q’s shape
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)  # (n_tokens, n_heads, q_mult, d_head)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)  # (n_tokens, n_heads, q_mult, d_head)

    # Reshape S (score modifier) to align with attention logits
    # Then expand across tokens so each head/q_mult has its own bias
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)

    # Create a causal mask (upper triangular with -inf above diagonal)
    # This prevents attending to future tokens
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)

    # If sliding_window > 0, also mask out tokens beyond the window size
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )

    # Compute attention scores: Q dot K^T
    # "qhmd,khmd->hmqk" means:
    #   q = query index, k = key index, h = head, m = q_mult, d = dimension
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)

    # Scale scores by sm_scale (usually 1/sqrt(d_head))
    QK *= sm_scale

    # Add mask to enforce causality/sliding window
    QK += mask[None, None, :, :]

    # Concatenate S (bias scores) along the last dimension
    QK = torch.cat([QK, S], dim=-1)

    # Apply softmax to get attention weights
    W = torch.softmax(QK, dim=-1)

    # Remove the extra bias dimension (last one from S)
    W = W[..., :-1]

    # Compute weighted sum of values: attention output
    # "hmqk,khmd->qhmd" means:
    #   multiply weights W with values V, sum over keys
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)

    # Reshape back to (n_tokens, n_heads * q_mult * d_head)
    return attn.reshape(n_tokens, -1)

"""

### 1. Inputs
- Queries: \(Q \in \mathbb{R}^{n_{tokens} \times n_{heads} \times q_{mult} \times d_{head}}\)  
- Keys: \(K \in \mathbb{R}^{n_{tokens} \times n_{heads} \times d_{head}}\)  
- Values: \(V \in \mathbb{R}^{n_{tokens} \times n_{heads} \times d_{head}}\)  
- Bias/score modifier: \(S\)  
- Scale factor: \(sm\_scale = \frac{1}{\sqrt{d_{head}}}\)  
- Sliding window size: `sliding_window`

---

### 2. Expand Keys and Values
We align them with the query multiplier dimension:
\[
K' = \text{expand}(K[:,:,None,:],\; q_{mult}) \quad \in \mathbb{R}^{n_{tokens} \times n_{heads} \times q_{mult} \times d_{head}}
\]
\[
V' = \text{expand}(V[:,:,None,:],\; q_{mult}) \quad \in \mathbb{R}^{n_{tokens} \times n_{heads} \times q_{mult} \times d_{head}}
\]

---

### 3. Reshape Bias
\[
S' = \text{reshape}(S,\; (n_{heads}, q_{mult}, 1, 1)) \quad \to \quad \text{expand to } (n_{heads}, q_{mult}, n_{tokens}, 1)
\]

---

### 4. Build Mask
- **Causal mask**:  
  \[
  M_{causal}[i,j] =
  \begin{cases}
  0 & j \leq i \\
  -\infty & j > i
  \end{cases}
  \]
- **Sliding window mask** (if `sliding_window > 0`):  
  additionally set \(M[i,j] = -\infty\) if \(i-j > sliding\_window\).

So final mask \(M \in \mathbb{R}^{n_{tokens} \times n_{tokens}}\).

---

### 5. Compute Raw Scores
Using Einstein summation:
\[
QK[i,h,m,q,k] = \sum_{d} Q[q,h,m,d] \cdot K'[k,h,m,d]
\]
Shape: \((n_{heads}, q_{mult}, n_{tokens}, n_{tokens})\).

---

### 6. Scale
\[
QK \leftarrow QK \cdot sm\_scale
\]

---

### 7. Apply Mask
\[
QK[i,h,m,q,k] \leftarrow QK[i,h,m,q,k] + M[q,k]
\]

---

### 8. Concatenate Bias
\[
QK' = \text{concat}(QK, S', \text{along last dim})
\]
Shape: \((n_{heads}, q_{mult}, n_{tokens}, n_{tokens}+1)\).

---

### 9. Softmax
\[
W = \text{softmax}(QK', \text{dim=-1})
\]

---

### 10. Remove Bias Slot
\[
W \leftarrow W[...,:-1] \quad \in \mathbb{R}^{n_{heads}, q_{mult}, n_{tokens}, n_{tokens}}
\]

---

### 11. Weighted Sum of Values
\[
attn[q,h,m,d] = \sum_{k} W[h,m,q,k] \cdot V'[k,h,m,d]
\]
Shape: \((n_{tokens}, n_{heads}, q_{mult}, d_{head})\).

---

### 12. Final Reshape
Flatten heads × q_mult × d_head:
\[
\text{Output} = \text{reshape}(attn,\; (n_{tokens}, n_{heads} \cdot q_{mult} \cdot d_{head}))
\]

---

### 🎯 Intuition Recap
1. Expand K/V → align with queries.  
2. Mask → enforce causality + optional sliding window.  
3. Dot product Q·K → attention scores.  
4. Scale → stabilize.  
5. Softmax → probabilities.  
6. Multiply with V → attended values.  
7. Reshape → final embeddings per token.

---


- \(n_{tokens}=3\)  
- \(n_{heads}=1\)  
- \(q_{mult}=1\)  
- \(d_{head}=2\)  

---

### 1. Define Q, K, V
```python
Q = torch.tensor([
    [[[1.0, 0.0]]],   # token 0
    [[[0.0, 1.0]]],   # token 1
    [[[1.0, 1.0]]]    # token 2
])  # shape (3,1,1,2)

K = torch.tensor([
    [[1.0, 0.0]],     # token 0
    [[0.0, 1.0]],     # token 1
    [[1.0, 1.0]]      # token 2
])  # shape (3,1,2)

V = torch.tensor([
    [[10.0, 0.0]],    # token 0
    [[0.0, 10.0]],    # token 1
    [[5.0, 5.0]]      # token 2
])  # shape (3,1,2)
```

---

### 2. Expand K, V
```python
K' = K[:,:,None,:].expand(-1,-1,1,-1)  # shape (3,1,1,2)
V' = V[:,:,None,:].expand(-1,-1,1,-1)  # shape (3,1,1,2)
```

---

### 3. Mask
Causal mask for 3 tokens:
\[
M =
\begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}
\]

---

### 4. Compute raw scores \(QK\)
\[
QK[q,k] = Q[q] \cdot K[k]^T
\]

- Token 0 query \([1,0]\):  
  vs K0 \([1,0]\) → 1  
  vs K1 \([0,1]\) → 0  
  vs K2 \([1,1]\) → 1  

- Token 1 query \([0,1]\):  
  vs K0 → 0  
  vs K1 → 1  
  vs K2 → 1  

- Token 2 query \([1,1]\):  
  vs K0 → 1  
  vs K1 → 1  
  vs K2 → 2  

So:
\[
QK =
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
1 & 1 & 2
\end{bmatrix}
\]

---

### 5. Scale
\(sm\_scale = 1/\sqrt{2} \approx 0.707\).  
Multiply each entry by 0.707.

---

### 6. Apply mask
Add \(-\infty\) where future tokens are blocked:

\[
QK =
\begin{bmatrix}
0.707 & -\infty & -\infty \\
0     & 0.707   & -\infty \\
0.707 & 0.707   & 1.414
\end{bmatrix}
\]

---

### 7. Softmax
Row‑wise softmax:

- Row 0 → [1,0,0]  
- Row 1 → [0.33,0.67,0]  
- Row 2 → [0.25,0.25,0.5]

---

### 8. Weighted sum with V
\[
attn[q] = \sum_k W[q,k] \cdot V[k]
\]

- Token 0: [10,0]  
- Token 1: 0.33*[10,0] + 0.67*[0,10] = [3.3, 6.7]  
- Token 2: 0.25*[10,0] + 0.25*[0,10] + 0.5*[5,5] = [5,5]

---

### ✅ Final Output
\[
\text{attn} =
\begin{bmatrix}
[10, 0] \\
[3.3, 6.7] \\
[5, 5]
\end{bmatrix}
\]

---

### 🎯 What you see
- Token 0 attends only to itself.  
- Token 1 attends to token 0 and itself.  
- Token 2 attends to all past tokens, weighted by similarity.  

That’s the **complete math pipeline**: Q·K → scale → mask → softmax → weighted sum with V → final attended representation.

---

"""
class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        # Dimension of each attention head
        self.head_dim = config.head_dim
        # Total number of attention heads
        self.num_attention_heads = config.num_attention_heads
        # Number of key/value heads (can be fewer than query heads in grouped attention)
        self.num_key_value_heads = config.num_key_value_heads

        # Apply sliding window attention only on even-numbered layers
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        # Learnable sink parameters (bias terms for attention scores)
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16))
        
        # nn.Parameter is a special type of tensor in PyTorch that is automatically registered as a learnable parameter of a module
# When you wrap a tensor with nn.Parameter, PyTorch will:
# 1. Automatically add it to the module's list of parameters (accessible via .parameters())
# 2. Include it in gradient computation during backpropagation
# 3. Update it during optimization (e.g., with SGD, Adam, etc.)
# 4. Save and load it when you save/load the model state
# Example: self.sinks = nn.Parameter(torch.empty(...)) creates a learnable parameter that will be optimized during training 
        # Normalization layer (RMSNorm instead of LayerNorm)
        self.norm = RMSNorm(config.hidden_size, device=device)

        # Dimension for Q, K, V projection:
        # queries = num_attention_heads * head_dim
        # keys + values = 2 * num_key_value_heads * head_dim
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )

        # Linear layer to project hidden states into Q, K, V
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )

        # Linear layer to project attention output back to hidden size
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )

        # Scaling factor for dot-product attention (1/sqrt(d_head))
        self.sm_scale = 1 / math.sqrt(config.head_dim)

        # Rotary positional embedding (RoPE) for injecting position information
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input
        t = self.norm(x)

        # Project into Q, K, V
        qkv = self.qkv(t)

        # Split Q, K, V from the concatenated projection
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()

        # Reshape Q, K, V into multi-head format
        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)

        # Apply rotary positional embeddings to Q and K
        q, k = self.rope(q, k)

        # Compute scaled dot-product attention with optional sliding window
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)

        # Project attention output back to hidden size
        t = self.out(t)

        # Residual connection: add original input
        t = x + t

        return t


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    """
    SwiGLU activation function with clamping.

    SwiGLU (Swish-Gated Linear Unit) is a variant of the GLU activation that
    combines a gating mechanism with the Swish function. It is often used in
    transformer feed-forward layers to improve training stability and model
    performance.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor. The last dimension is split into two halves:
        - even indices (x_glu) for the gating branch
        - odd indices (x_linear) for the linear branch
    alpha : float, default=1.702
        Scaling factor applied inside the sigmoid for the Swish gate.
        (1.702 is a common choice to approximate GELU behavior.)
    limit : float, default=7.0
        Clamping limit to prevent extreme values and improve numerical stability.

    Returns
    -------
    torch.Tensor
        Output tensor after applying SwiGLU activation:
        out = (x_glu * sigmoid(alpha * x_glu)) * (x_linear + 1)

    Notes
    -----
    - The input is split into two parts: one controls the gate (x_glu),
      the other is the linear branch (x_linear).
    - Clamping ensures values don’t explode during training.
    - An extra bias of +1 is added to the linear branch before multiplication.
    """

    # Split input into two halves along the last dimension
    x_glu, x_linear = x[..., ::2], x[..., 1::2]

    # Clamp values to avoid numerical instability
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)

    # SwiGLU gate: swish(x_glu) = x_glu * sigmoid(alpha * x_glu)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)

    # Multiply gated branch with linear branch (+1 bias)
    return out_glu * (x_linear + 1)

class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        # Number of experts in the mixture-of-experts (MoE) layer
        self.num_experts = config.num_experts
        # How many experts each token is allowed to use (top-k routing)
        self.experts_per_token = config.experts_per_token
        # Clamp limit for the SwiGLU activation
        self.swiglu_limit = config.swiglu_limit

        # World size for distributed training (default 1 if not using distributed)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Normalization layer before gating
        self.norm = RMSNorm(config.hidden_size, device=device)

        # Gating network: projects hidden states to expert scores
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )

        # Ensure intermediate size divides evenly across distributed workers
        assert config.intermediate_size % self.world_size == 0
        
        # Build experts as separate modules (each expert is a small MLP)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                # First linear layer expands hidden size → intermediate size * 2
                torch.nn.Linear(
                    config.hidden_size, 
                    config.intermediate_size * 2 // self.world_size, 
                    device=device, 
                    dtype=torch.bfloat16
                ),
                # Second linear layer projects back to hidden size
                torch.nn.Linear(
                    config.intermediate_size // self.world_size, 
                    config.hidden_size, 
                    device=device, 
                    dtype=torch.bfloat16
                )
            ) for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (seq_len, hidden_size)
        seq_len, hidden_size = x.shape

        # Normalize input
        t = self.norm(x)

        # Compute gating scores for each expert
        g = self.gate(t)

        # Select top-k experts per token
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)  # normalize weights
        expert_indices = experts.indices  # which experts were chosen

        # Flatten for easier processing
        t_flat = t.view(-1, hidden_size)
        expert_indices_flat = expert_indices.view(-1, self.experts_per_token)
        expert_weights_flat = expert_weights.view(-1, self.experts_per_token)

        # Initialize output buffer
        output = torch.zeros_like(t_flat)

        # Process tokens for each expert
        for expert_idx in range(self.num_experts):
            # Mask tokens that selected this expert
            mask = (expert_indices_flat == expert_idx).any(dim=-1)
            if not mask.any():
                continue

            # Get token indices routed to this expert
            token_indices = torch.where(mask)[0]
            # Position of this expert in the top-k list
            expert_pos = (expert_indices_flat[token_indices] == expert_idx).nonzero(as_tuple=True)[1]

            # Gather inputs and weights for this expert
            expert_input = t_flat[token_indices]
            weights = expert_weights_flat[token_indices, expert_pos]

            # Forward pass through expert MLP
            expert_out = expert_input
            expert_out = self.experts[expert_idx][0](expert_out)  # First linear
            expert_out = swiglu(expert_out, limit=self.swiglu_limit)  # SwiGLU activation
            expert_out = self.experts[expert_idx][1](expert_out)  # Second linear

            # Weighted contribution added to output
            output[token_indices] += expert_out * weights.unsqueeze(-1)

        # Aggregate across distributed workers if needed
        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)

        # Reshape back to (seq_len, hidden_size)
        output = output.view(seq_len, hidden_size)

        # Residual connection: add original input
        return x + output


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x

class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        """
        Transformer model composed of stacked TransformerBlocks.

        Parameters
        ----------
        config : ModelConfig
            Configuration object containing model hyperparameters
            (vocab size, hidden size, number of layers, etc.).
        device : torch.device, optional
            Device to place the model on (CPU/GPU).

        Components
        ----------
        embedding : nn.Embedding
            Maps token IDs to dense hidden vectors.
        block : nn.ModuleList
            Sequence of TransformerBlock layers (attention + MLP).
        norm : RMSNorm
            Final normalization before output projection.
        unembedding : nn.Linear
            Projects hidden states back to vocabulary logits.
        """
        super().__init__()
        # Token embedding layer
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )

        # Stack of Transformer blocks
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, device=device)

        # Output projection back to vocabulary space
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of token IDs (shape: [seq_len]).

        Returns
        -------
        torch.Tensor
            Logits over vocabulary for each token position.
        """
        # Convert token IDs to embeddings
        x = self.embedding(x)

        # Pass through each Transformer block
        for block in self.block:
            x = block(x)

        # Normalize
        x = self.norm(x)

        # Project back to vocabulary logits
        x = self.unembedding(x)
        return x

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "Transformer":
        """
        Load a Transformer model from a checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint directory containing config.json and weights.
        device : str or torch.device, default="cuda"
            Device to load the model onto.

        Returns
        -------
        Transformer
            Model instance loaded with configuration and weights.
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # Load configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        # Initialize model
        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()

        # NOTE: Weight loading logic is commented out here.
        # In practice, this section would shard and load parameters
        # across distributed workers.

        return model



class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        """
        TokenGenerator wraps a Transformer model for autoregressive token generation.

        Parameters
        ----------
        checkpoint : str
            Path to the model checkpoint directory.
        device : torch.device
            Device on which to load and run the model (CPU/GPU).

        Notes
        -----
        - Uses inference mode to disable gradient tracking for efficiency.
        - Loads a Transformer model from the given checkpoint.
        """
        self.device = device
        # Load the Transformer model from checkpoint
        self.model = Transformer.from_checkpoint("./", device=self.device)

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        """
        Generate tokens autoregressively from a prompt.

        Parameters
        ----------
        prompt_tokens : list[int]
            Initial sequence of tokens to start generation.
        stop_tokens : list[int]
            List of tokens that will terminate generation if produced.
        temperature : float, default=1.0
            Sampling temperature. 
            - 0.0 → greedy decoding (argmax).
            - >0.0 → probabilistic sampling with softmax.
        max_tokens : int, default=0
            Maximum number of tokens to generate. 
            - 0 means unlimited until a stop token is encountered.
        return_logprobs : bool, default=False
            If True, yields (token, logprob) pairs instead of just tokens.

        Yields
        ------
        int or (int, float)
            Generated token IDs, optionally with log-probabilities.

        Notes
        -----
        - Autoregressive loop: each new token is appended and fed back into the model.
        - Stops when either a stop token is generated or max_tokens is reached.
        """
        tokens = list(prompt_tokens)
        num_generated_tokens = 0

        # Loop until max_tokens reached or stop token encountered
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            # Forward pass: get logits for the next token
            logits = self.model(torch.as_tensor(tokens, dtype=torch.int32, device=self.device))[-1]

            # Choose next token
            if temperature == 0.0:
                # Greedy decoding (argmax)
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                # Sample from softmax distribution
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

            # Append token to sequence
            tokens.append(predicted_token)
            num_generated_tokens += 1

            # Yield token (and logprob if requested)
            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            # Stop if a stop token is generated
            if predicted_token in stop_tokens:
                break



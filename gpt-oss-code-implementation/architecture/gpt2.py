import torch 
import torch.nn as nn
from torch.nn import functional as F

GPT_CONFIG = {
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_head': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': True,
}


class LayerNorm(nn.Module):
    """
    Layer normalization module that normalizes the input tensor across the last dimension. It computes the mean and variance of the input tensor, normalizes it, and applies learnable scaling and bias parameters. The normalization is done using the formula:
    norm = (x - mean) / sqrt(var + eps)
    why we are taking the last dimension for normalization is that in many deep learning architectures, especially in transformer models, the last dimension often represents the feature dimension (e.g., embedding size). Normalizing across this dimension helps stabilize the learning process and improves convergence. By normalizing across the feature dimension, we ensure that each feature contributes equally to the model's output, preventing any single feature from dominating the learning process. This is particularly important in models like GPT-2, where the input consists of sequences of tokens represented as embeddings.
    """
    def __init__(self, embed, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed))
        self.bias = nn.Parameter(torch.zeros(embed))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim = -1,keepdim=True) # mean -> dim = -1, keepdim = True where dim is the last dimension of the input tensor x. This computes the mean across the last dimension while keeping the same number of dimensions in the output tensor. Example: If x has shape (batch_size, seq_length,embed_dim), then mean will have shape (batch_size, seq_length, 1) where 1 is the size of the last dimension after computing the mean. 
        var = x.var(dim = -1,unbiased=False,keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return norm * self.weight + self.bias 


class GLUE(nn.Module):
    """ 
    GLUE - Gated Linear unit  (A(x)) @ sigma(B(X)) where A, B are leaned wwight matrices and sigma is gating function @ denotes elementwise multiplication
    it uses gated mechanism like a sigmoid to control how much of one linear projection pases through ,multiplying it elementwise with another projection 

    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x=0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
        return x

class FeedForward(nn.Module):
    """
    FeedForward module that consists of two linear layers with a GELU activation function in between. It takes an input tensor, applies a linear transformation, passes it through the GELU activation, and then applies another linear transformation. This module is commonly used in transformer architectures to introduce non-linearity and increase the model's capacity to learn complex representations.
    """
    def __init__(self, embed, ff_mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed, ff_mult * embed),
            GLUE(),
            nn.Linear(ff_mult * embed, embed)
        )
    def forward(self, x):
        return self.net(x)

    
class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention module that implements the multi-head self-attention mechanism. It takes an input tensor, projects it into query, key, and value tensors, splits them into multiple heads, computes scaled dot-product attention for each head, and then concatenates the results. This module allows the model to focus on different parts of the input sequence simultaneously, capturing various relationships and dependencies.
    
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0) , "d_out must be divisible by num_heads"
        self.d_out = d_out 
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # reduce the projetion dim to match the desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in,d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in,d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer( 
            "mask", # register_buffer is to register a tensor as a persistent buffer in the module, which means it will be saved and loaded with the model's state_dict but won't be considered a model parameter (i.e., it won't be updated during training). This is useful for tensors that are part of the model's state but don't require gradients, such as masks or fixed constants.
            torch.tril(torch.ones(context_length, context_length) , diagonal = 1) # torch.triu returns the upper triangular part of a matrix, while torch.tril returns the lower triangular part. In the context of attention mechanisms, we often use a lower triangular mask to prevent the model from attending to future tokens in a sequence. This is important for autoregressive models like GPT-2, where the prediction for a token should only depend on the tokens that come before it in the sequence. By using torch.tril, we create a mask that allows each token to attend only to itself and the tokens that precede it, effectively enforcing causality in the attention mechanism ,where diagonal = 1 means that the diagonal elements are included in the lower triangular part, allowing each token to attend to itself as well as previous tokens. The resulting mask will have a shape of (context_length, context_length), where context_length is the maximum sequence length that the model can handle. The mask will contain 1s in the lower triangular part and 0s in the upper triangular part, indicating which positions are allowed to attend to which other positions during the attention computation.
        )
    def forward(self,x):
        b, num_tokens, d_in  = x.shape # b , num_tokens,d_in - batch, no. of tokens, input dimensions
        keys = self.W_key(x) # shape : (b,num_tokens,d_out)
        queries = self.W_query
        values = self.W_value

        # split the matrix by adding a num_heads dimensions
        # unroll the last dimension : (b,num_tokens,d_out) - > (b, num_tokens,num_heads,head_dim) where num head defines how many attention heads we want to use and head_dim is the dimension of each head. This allows the model to learn different representations for different parts of the input sequence, capturing various relationships and dependencies.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # shape : (b,num_heads,num_tokens,head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim) tranposing because we want to perform attention for each head separately, and having the num_heads dimension before the num_tokens dimension allows us to easily compute attention scores for each head independently. This arrangement is more efficient for parallel computation and aligns with the way multi-head attention is typically implemented in transformer architectures.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
    
        # compute the scaled dot product attention with a causal mask to prevent attending to future tokens. The attention scores are computed by taking the dot product of the queries and keys, scaling them by the square root of the head dimension, and applying the mask to ensure that each token can only attend to itself and previous tokens. The softmax function is then applied to obtain the attention weights, which are used to compute a weighted sum of the values. Finally, the output is projected back to the original dimension using a linear layer. 
        attn_scores = queries @ keys.transpose(2,3) # dot proecut of each head where tranpose (2,3) is to transpose the last two dimensions of the keys tensor, which represent the sequence length and head dimension. This allows us to compute the dot product between the queries and keys for each head independently, resulting in a tensor of attention scores with shape (b, num_heads, num_tokens, num_tokens). The attention scores indicate how much each token should attend to every other token in the sequence for each head.
        
        #original mask truncated to the number of tokens and convert to boolean 
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # shape : (num_tokens,num_tokens) where we are truncating the mask to match the number of tokens in the input sequence. This is important because the input sequence may be shorter than the maximum context length, and we want to ensure that the attention mechanism only considers valid token positions. By slicing the mask, we create a boolean mask that indicates which positions are allowed to attend to which other positions during the attention computation.

        # use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights  = torch.softmax(attn_scores / keys.shape[-1]**0.5,dim = -1)

        attn_weights = self.dropout(attn_weights)

        #shape (b,num_tokens,nu_heads,head_dim)
        context_vec = (attn_weights @ values).transpose(1,2) # tranpose 1,2 to bring the num_heads dimension back to its original position, resulting in a tensor of shape (b, num_tokens, num_heads, head_dim). This allows us to concatenate the outputs from all attention heads before projecting them back to the original dimension.

        #combine heads , where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection
        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ln1= LayerNorm(embd=cfg["emb_dim"])
        self.ln2= LayerNorm(embd=cfg["emb_dim"])
        self.ff= FeedForward(cfg)
        self.dropout= nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x):
        shortcut=x
        x= self.ln1(x)
        x=self.attn(x)
        x=self.dropout(x)
        x= x+shortcut
        shortcut =x
        x=self.ln2(x)
        x=self.ff(x)
        x=self.dropout(x)
        x=x+shortcut
        return x
    
    
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg= cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
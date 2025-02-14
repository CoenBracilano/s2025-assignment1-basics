from typing import Optional
import torch
import torch.nn as nn
import math


def gelu_func(x = torch.FloatTensor) -> torch.FloatTensor:
    return 0.5  * x.cpu() * (1 + torch.erf(x / (math.sqrt(2))))


class PWFF(nn.Module):
    def __init__(self, d_model: int, d_ff: int, weights: dict[str, torch.FloatTensor]):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        #Initialize learnable weights 
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

        # Ensure required weight keys exist
        assert "w1.weight" in weights, "Missing 'w1.weight' in weights dictionary."
        assert "w2.weight" in weights, "Missing 'w2.weight' in weights dictionary."

        # Assign provided weights to the layers
        self.fc1.weight.data = weights["w1.weight"]
        self.fc2.weight.data = weights["w2.weight"]

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        #Computes w2*gelu(x*w1)
        return self.fc2(gelu_func(self.fc1(x)))

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, weights: dict[str, torch.FloatTensor]):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        

        assert "weight" in weights, "Missing 'weight' key in weights dictionary." #Make sure we have a weight tensor
        #Initialize a learnable weight parameter with our provided weights
        self.weight = nn.Parameter(weights["weight"])
        #Reminder weights are gi in our document

    def forward(self, input: torch.FloatTensor)-> torch.FloatTensor:
        #Calculate RMS on the last dimension
        #torch.mean infers d_model from input size 
        #dim=-1 internally does 1/d_model 
        rms = torch.sqrt(torch.mean(input**2, dim=-1, keepdim=True)+ self.eps)
        #Normalize our input
        normal = input / rms
        #Multiply by gi aka apply the weights to our normalized values
        return normal * self.weight


def softmax(input: torch.FloatTensor, dim: int)-> torch.FloatTensor:
    #c is our value for finding numerical stability
    #It is written as oi in the document 
    c = torch.max(input, dim=dim, keepdim=True).values
    scaled = input - c
    exp_vals = torch.exp(scaled)
    return exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)
 
def scaled_dot_product_attention(
    K: torch.FloatTensor, Q: torch.FloatTensor, 
    V: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None, 
    pdrop: Optional[float] = None)-> torch.FloatTensor:
    #Do dot product of Q and K^T
    dot = Q @ (K.transpose(-2,-1))

    #Find the denominator sqrt(dimension of K)
    denom = math.sqrt(K.shape[-1])

    #Compute division
    div = dot / denom

    #If we have a mask, check if our dimensions match and rescale mask to make it fit div
    #Then go through and apply the mask, setting any true values = -inf
    if mask is not None:
        while mask.dim() < div.dim():
            mask = mask.unsqueeze(0)  # Expand dimensions to match batch shape
        div = div.masked_fill(mask, float("-inf"))
    
    #Apply softmax to get a prob dist
    soft = torch.softmax(div, dim=-1)

    #If we have a a dropout probabilty, randomly dropout values based on the given probability
    if pdrop is not None:
        soft = torch.nn.functional.dropout(soft, p=pdrop, training=True)
    #Scale output by Values

    return soft @ V

class multiHeadAttn(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float, weights: dict[str, torch.FloatTensor]):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_head = d_model // num_heads #Find the dimensionality of each head

        #Extract weights from the weight list
        q_weights = torch.stack([weights[f"q_heads.{i}.weight"] for i in range(num_heads)], dim=0)  # (num_heads, d_head, d_model)
        k_weights = torch.stack([weights[f"k_heads.{i}.weight"] for i in range(num_heads)], dim=0)  # (num_heads, d_head, d_model)
        v_weights = torch.stack([weights[f"v_heads.{i}.weight"] for i in range(num_heads)], dim=0)  # (num_heads, d_head, d_model)

        # Combine the weights into a single weight matrix to make computation more efficiant
        self.qkv_proj = torch.cat([q_weights, k_weights, v_weights], dim=0)
        self.qkv_proj = nn.Parameter(self.qkv_proj.reshape(3 * self.num_heads * self.d_head, d_model))

        self.output_proj = nn.Parameter(weights["output_proj.weight"])


    def forward(self, input: torch.FloatTensor)-> torch.FloatTensor:
        #The _ is used as a throwaway variable that we wont access again 
        #Shape returns three values and we only want two for the time being
        batch_size, seq_length, _ = input.shape

        # Compute the weights accross the input using the combined weight matrix 
        qkv = torch.matmul(input, self.qkv_proj.T)
        # Reshape our tensor
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.d_head)
        #After we have computed our multiplcation we split our matrix back into queries keys and values
        queries, keys, values = qkv.unbind(dim=2)

        #Tanspose our matricies 
        queries, keys, values = queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3)
        
        #Create a mask of ones in an upper trianglular matrix for the size of the sequence length
        #Also check that we are creating it on the right deivce for the input
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(input.device)
        #Reshape our mask 
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        #Compute attention using our scaled dot product function
        attn_output = scaled_dot_product_attention(keys, queries, values, mask, self.attn_pdrop)
        
        #Concatinate our heads 
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, -1)

        #Apply the output projection
        output = torch.matmul(attn_output, self.output_proj.T)
        return output



class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 attn_pdrop: float, residual_pdrop: float, 
                 weights: dict[str, torch.FloatTensor]):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.d_head = d_model // num_heads  # Dimension of each head

        # Extract weights
        self.q_proj_weight = weights["attn.q_proj.weight"]
        self.k_proj_weight = weights["attn.k_proj.weight"]
        self.v_proj_weight = weights["attn.v_proj.weight"]
        self.output_proj_weight = weights["attn.output_proj.weight"]

        self.ln1_weight = weights["ln1.weight"]
        self.ffn_w1_weight = weights["ffn.w1.weight"]
        self.ffn_w2_weight = weights["ffn.w2.weight"]
        self.ln2_weight = weights["ln2.weight"]

        # Reshape key, query, value weights
        self.q_proj_weight = self.q_proj_weight.view(num_heads, self.d_head, d_model)
        self.k_proj_weight = self.k_proj_weight.view(num_heads, self.d_head, d_model)
        self.v_proj_weight = self.v_proj_weight.view(num_heads, self.d_head, d_model)
        self.shaped_weights = transform_weights(weights, num_heads)

        # Define the layers
        self.rms_norm1 = RMSNorm(d_model, 1.0e-5, {"weight": self.ln1_weight})
        self.rms_norm2 = RMSNorm(d_model, 1.0e-5, {"weight": self.ln2_weight})

        self.multihead_attn = multiHeadAttn(d_model, num_heads, attn_pdrop, self.shaped_weights)
        ffn_weights = {
            "w1.weight": self.ffn_w1_weight,
            "w2.weight": self.ffn_w2_weight
        }
        self.pwff = PWFF(d_model, d_ff, ffn_weights)

        # Define dropout layers
        self.dropout_residual = nn.Dropout(residual_pdrop)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply first RMSNorm
        rms_out = self.rms_norm1(x)

        # Multi-Head Attention
        attn_out = self.multihead_attn(rms_out)

        # Apply dropout and residual connection
        attn_out = self.dropout_residual(attn_out)
        first_sum = x + attn_out  # Residual connection

        # Apply second RMSNorm
        rms_out = self.rms_norm2(first_sum)

        # Feed-forward network
        pwff_out = self.pwff(rms_out)

        # Apply dropout and final residual connection
        pwff_out = self.dropout_residual(pwff_out)
        return first_sum + pwff_out


def transform_weights(weights, num_heads):
    transformed_weights = {}

    # Extract large concatenated weights
    q_proj_weight = weights["attn.q_proj.weight"]
    k_proj_weight = weights["attn.k_proj.weight"]
    v_proj_weight = weights["attn.v_proj.weight"]
    output_proj_weight = weights["attn.output_proj.weight"]

    # Split into individual heads
    q_heads = torch.chunk(q_proj_weight, num_heads, dim=0)
    k_heads = torch.chunk(k_proj_weight, num_heads, dim=0)
    v_heads = torch.chunk(v_proj_weight, num_heads, dim=0)

    # Store the split weights
    for i in range(num_heads):
        transformed_weights[f"q_heads.{i}.weight"] = q_heads[i]
        transformed_weights[f"k_heads.{i}.weight"] = k_heads[i]
        transformed_weights[f"v_heads.{i}.weight"] = v_heads[i]

    # Store the output projection weight
    transformed_weights["output_proj.weight"] = output_proj_weight

    return transformed_weights



class Transformer_LM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int,
                 d_ff: int, attn_pdrop: float, residual_pdrop: float, weights: dict[str, torch.FloatTensor]):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.residual_pdrop = residual_pdrop

        # Initialize token and position embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.token_embeddings.weight = nn.Parameter(weights['token_embeddings.weight'])
        self.position_embeddings.weight = nn.Parameter(weights['position_embeddings.weight'])

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                attn_pdrop=attn_pdrop,
                residual_pdrop=residual_pdrop,
                weights=self._extract_layer_weights(weights, layer_idx)
            )
            for layer_idx in range(num_layers)
        ])

        # RMSNorm
        self.rms = RMSNorm(d_model=self.d_model, eps=1.0e-5, weights={"weight": weights['ln_final.weight']})

        # Output linear layer (language model head)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(weights['lm_head.weight'])

        #Initilize dropout
        self.dropout = nn.Dropout(residual_pdrop)

    def _extract_layer_weights(self, weights: dict[str, torch.FloatTensor], layer_idx: int) -> dict[str, torch.FloatTensor]:
        """ Extract only the weights relevant to a specific TransformerBlock. """
        return {
            'attn.q_proj.weight': weights[f'layers.{layer_idx}.attn.q_proj.weight'],
            'attn.k_proj.weight': weights[f'layers.{layer_idx}.attn.k_proj.weight'],
            'attn.v_proj.weight': weights[f'layers.{layer_idx}.attn.v_proj.weight'],
            'attn.output_proj.weight': weights[f'layers.{layer_idx}.attn.output_proj.weight'],
            'ln1.weight': weights[f'layers.{layer_idx}.ln1.weight'],
            'ffn.w1.weight': weights[f'layers.{layer_idx}.ffn.w1.weight'],
            'ffn.w2.weight': weights[f'layers.{layer_idx}.ffn.w2.weight'],
            'ln2.weight': weights[f'layers.{layer_idx}.ln2.weight'],
        }

    def forward(self, input: torch.LongTensor) -> torch.FloatTensor:
        batch_size, seq_len = input.shape

        # Token and position embeddings
        token_emb = self.token_embeddings(input)  
        positions = torch.arange(seq_len, device=input.device).expand(batch_size, seq_len)
        pos_emb = self.position_embeddings(positions)

        # Sum token and position embeddings
        current_value = token_emb + pos_emb

        # Apply dropout 
        current_value = self.dropout(current_value)

        # Transformer blocks
        for layer in self.layers:
            current_value = layer(current_value)

        # Apply RMSNorm
        current_value = self.rms(current_value)

        # Linear projection
        current_value = self.lm_head(current_value)

        # Apply softmax (using your function)
        return current_value
        
if __name__ == "__main__":

    input_path = "test"
    d_model = 512
    provided_weights = {"weight": torch.ones(d_model)}
    eps: float = 1.0e-5
    rms_norm = RMSNorm(d_model, eps, provided_weights)
    x = torch.randn(2, 4, d_model)
    output = rms_norm(x)
    print(output.shape)
    d_ff = 2048
    pwff = PWFF(d_model, d_ff, provided_weights)
    out = pwff(x)

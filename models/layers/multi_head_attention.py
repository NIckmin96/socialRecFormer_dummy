import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Perform scaled dot product attention
    """
    def __init__(self, is_enc=False):
        super(ScaledDotProductAttention, self).__init__()
        if is_enc:
            self.spd_param = nn.Parameter(torch.randn((30, 30), dtype=torch.float, requires_grad=True))
    
    def forward(self, Q, K, V, mask=None, attn_bias=None, last_layer_flag=False, is_dec_layer=False, is_rating=True):
        # Input is 4-d tensor
        batch_size, head, length, d_tensor = K.size()

        # 1. Compute similarity by Q.dot(K^T)
        K_T = K.transpose(2, 3)
        score = torch.matmul(Q, K_T) / math.sqrt(d_tensor)
        batch_size, head, len_a, len_b = score.size() # enc : len_a(user), len_b(user) / dec : len_a(item), len_b(user)

        # 2. Apply attention mask
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000) # mask의 값이 0인 위치에 해당하는 attention score값을 -10000으로 변경

        # 3. Apply attention bias (spatial encoding)
        loss = 0
        
        ######################################################## [ORG] ########################################################
        if attn_bias is not None:
            # score += attn_bias
            if is_dec_layer: 
                if is_rating:
                    # Rating loss(RMSE)
                    loss = torch.sqrt(F.mse_loss(score.float(), attn_bias.float(), reduction='mean'))
                else:
                    # Ranking loss(BCE)
                    loss = F.binary_cross_entropy_with_logits(score.squeeze(-1).float(), attn_bias.float(), reduction='mean')
    
            else:
                # encoder loss(attn_bias : user간의 distance / encoder attention score vs attn_bias)
                attn_bias = torch.where(attn_bias == 0, 1.0, (1/(attn_bias)**2).double())
                loss = torch.sqrt(F.mse_loss(score.float(), attn_bias.float(), reduction='mean')) # [TODO] MSE loss에 대해서 다시 sqrt를 취하고, element의 개수로 나눠서 loss를 계산하는게 맞는지?
                
        ### Decoder 마지막 layer에서 Q * K.T(score)의 Head를 기준으로한 mean값을 Return
        if last_layer_flag:
            score = torch.mean(score, dim=1)
            return score, loss

        # 3. Pass score to softmax for making [0, 1] range.
        score = torch.softmax(score, dim=-1)

        # 4. Dot product with V
        V = torch.matmul(score, V)
        
        return V, loss

class MultiHeadAttention(nn.Module):
    """
    Perform multi-head attention
    """
    def __init__(self, d_model, num_heads, last_layer_flag=False, is_dec_layer=False, is_rating=True):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(not is_dec_layer)
        self.last_layer_flag = last_layer_flag
        self.is_dec_layer = is_dec_layer
        self.is_rating = is_rating

        # Input projection
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_concat = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None, attn_bias=None):
        
        # 1. Dot produt with weight matrices
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)

        # 2. Split tensor by number of heads
        Q, K, V = self.split(Q), self.split(K), self.split(V)

        # Apply mask for multi-head attention
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        if not self.last_layer_flag:
            # 3. Perform scaled-dot product attention
            out, loss = self.attention(Q, K, V, mask, attn_bias, self.last_layer_flag, self.is_dec_layer, self.is_rating)
            
        # last layer : Decoder의 마지막 layer (cross-attn)는 rating prediction을 수행
        else: 
            out, loss = self.attention(Q, K, V, mask, attn_bias, self.last_layer_flag, self.is_dec_layer, self.is_rating)
            return out, loss
        
        # 4. Concat and pass to linear layer
        out = self.concat(out)
        out = self.W_concat(out)

        return out, loss
    
    def split(self, tensor):
        """
        Split tensor by number of heads

        Input tensor shape: 
            (batch_size, length, d_model)
        Outout tensor shape:
            (batch_size, num_head, length, d_tensor)
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.num_heads
        tensor = tensor.view(batch_size, length, self.num_heads, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Inverse function of self.split()

        Input tensor shape:
            (batch_size, num_head, length, d_tensor)
        Output tensor shape:
            (batch_size, length, d_model)
        """
        batch_size, num_head, length, d_tensor = tensor.size()
        
        d_model = num_head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)

        return tensor
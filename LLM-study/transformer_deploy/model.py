import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim: int = 8, 
        head_num: int = 4, 
        dropout_rate: float = 0.2,
        is_masked: bool = False
        ):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // head_num
        self.head_num = head_num
        
        # A = (hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.dropout_rate = dropout_rate
        
        # This part is to decide weather it's for the decoder
        self.is_masked = is_masked
        
    def forward(self, x, mask = None):
        # x => (b, s, h)
        b, s, _ = x.shape
        
        # (b, s, h) -Linear-> (b, s, h)
        # (h, h)^T @ (b, s, h) + b = (h, h) @ (b, s, h) = (b, s, h)
        self.Q = self.q(x)
        self.K = self.q(x)
        self.V = self.q(x)
        
        # (b, s, h) -view-> (b, s, head_num, head_dim) -transpose-> (b, head_num, s, head_dim)
        Q_state = self.Q.view(b, s, self.head_num, self.head_dim)
        K_state = self.K.view(b, s, self.head_num, self.head_dim)
        V_state = self.V.view(b, s, self.head_num, self.head_dim)
        
        # (b, head_num, s, head_dim) @ (b, head_num, head_dim, s) = (b, head_num, s, s)
        atten_weight = Q_state @ K_state.transpose(-1, -2)
        atten_weight = atten_weight / math.sqrt(self.head_dim)
        if mask is not None:
            atten_weight = atten_weight
        
        # (b, head_num, s, s)
        atten_weight = torch.softmax(atten_weight, -1)
        atten_weight = self.dropout(atten_weight)
        
        # (b, head_num, s, s) @ (b, head_num, s, head_dim) = (b, head_num, s, head_dim) 
        output  = atten_weight @ V_state
        # -transpose(1,2)-> (b, s, head_num, head_dim) -contiguous-view(b, s, -1)-> 
        # (b, s, hidden_dim)
        output = output.contiguous().view(b, s, -1)
        
        # (b, s, hidden_dim) -Linear-> (b, s, s)
        output = self.out(output)
        
        return output

class FeedForwardLayer(nn.Module):
    '''
    This part is for forward and norm.
    '''
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        
        # (hidden_dim, hidden_dim * 4)
        self.up = nn.Linear(hidden_dim, hidden_dim * 4)
        self.up = nn.ReLU(self.up) 
        # (hidden_dim * 4, hidden_dim)
        self.down = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-7)
    
    def forward(self, x):
        # (h, 4h)^T @ (b, s, h) 
        # = (4h, h) @ (b, s, h)
        # = (b, s, 4h)
        up = self.up(x)
        down = self.down(up)
        return self.norm(x + down)

class EnCoderLayer(nn.Module):
    def __init__(self, hidden_dim: int = 12, dropout_rate: float = 0.2):
        super().__init__()
    
    def forward(self, q, k, x):
        return x

class DeCoderLayer(nn.Module):
    def __init__(self, hidden_dim: int = 12, dropout_rate: float = 0.2):
        super().__init__()
        
    def forward(self, x):
        return x

class EnCoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass

class DeCoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
if __name__ == "__main__":
    mmh = MultiHeadAttentionLayer(8, 4, 0.2)
    data = torch.randint(low=1, high=9, size=(1, 4, 8))
    print(data)
    output = mmh(data)
    print(output.shape)
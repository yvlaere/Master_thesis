class MultiHeadAttention(nn.Module):
    def __init__(self, heads, channels):#, dropout_prob = 0.1):
        super().__init__()
        
        self.heads = heads
        self.cph = channels // heads #afgeronde breuk, channels per head
        
        #usually features_in = features_out
        self.query = nn.Linear(channels, self.heads * self.cph)
        self.key = nn.Linear(channels, self.heads * self.cph)
        self.value = nn.Linear(channels, self.heads * self.cph)
    
    def self_attention(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.cph)   #scores: (batch x heads x seq_len x seq_len) (seq_len x seq_len is attention matrix)
        scores.masked_fill_(mask, -1e9)                                     #fills elements of self tensor with value where mask is True, value isn't -inf to keep softmax functional
        attn = nn.Softmax(dim = -2)(scores)
        context = torch.matmul(attn, V)
        
        #delete large tensors to free up space
        #del scores
        #del attn
        #del mask
        #torch.cuda.empty_cache()
        
        return context#, attn
    
    def forward(self, x, mask):
        #shape of Q, K and V: (Batch, Channels, Seq_len), here Q = K = V = data that goes into the multi head attention
        batch_size = x.shape[0]

        #prepare query, key, value through linear transformation
        # (B, C, S) -trans-> (B, S, C) -proj-> (B, S, C) -split-> (B, S, Heads, Cph) -trans-> (B, H, S, Cph)
        Q = self.query(x.transpose(1, 2)).view(batch_size, -1, self.heads, self.cph).transpose(1,2)     #Q: (batch x heads x seq_len x cph)
        K = self.key(x.transpose(1, 2)).view(batch_size, -1, self.heads, self.cph).transpose(1,2)       #K: (batch x heads x seq_len x cph)
        V = self.value(x.transpose(1, 2)).view(batch_size, -1, self.heads, self.cph).transpose(1,2)     #V: (batch x heads x seq_len x cph)

        #masks per sequence (batch element) are being duplicated for all heads, this happens here, because the amount of heads can differ between transformer blocks
        #mask (batch_size x seq_len x seq_len) -> (batch_size x n_heads x seq_len x seq_len)
        mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)     # mask : (batch x heads x seq_len x seq_len)

        #self attention
        z = self.self_attention(Q, K, V, mask)  #x: (batch x heads x seq_len x cph), identical to Q, K and V
        
        #does this help?
        #del mask
        #torch.cuda.empty_cache()
        #print("test")
        #print(x.shape)
        #print(z.shape)
        
        #merge cph and heads
        z = z.transpose(1, 2).reshape(batch_size, -1, self.heads * self.cph) #x: (batch x seq_len x heads * cph) = (B, S, C)
        #print(z.shape)
        return z
    

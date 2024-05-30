from config import *
import torch.nn as nn

import math
import torch
from mask import *


# 位置编码
class pos_embedding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_token=5000):
        super(pos_embedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_token, embedding_size) #torch.Size([5000, 512])
        position = torch.arange(0, max_token, dtype=torch.float).unsqueeze(1) #torch.Size([5000, 1])
        div_term = torch.exp(torch.arange(
            0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0).transpose(0, 1) #torch.Size([5000, 1, 512]) 最长5000个单词 每个单词编码512维向量
        self.register_buffer('pe', pe)

    def forward(self, x): #torch.Size([11, 2, 512])
        """
        x: [token_len, batch_size, embedding_size]
        """
        x = x + self.pe[:x.size(0), :] #torch.Size([5000, 1, 512]) #torch.Size([11, 2, 512])
        return self.dropout(x)
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        #torch.Size([2, 8, 11, 64]) #torch.Size([2, 8, 64, 11])
        scores = torch.matmul(Q, K.transpose(-1, -2)) / \
            np.sqrt(dimension_k) 
        scores.masked_fill_(attn_mask, -1e9) #torch.Size([2, 8, 11, 11])
        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax torch.Size([2, 8, 11, 11])
        context = torch.matmul(attn, V) #torch.Size([2, 8, 11, 11]) torch.Size([2, 8, 11, 64])-torch.Size([2, 8, 11, 64])
        return context, attn #torch.Size([2, 8, 11, 64]) torch.Size([2, 8, 11, 11])

# kqv点× 全连接
class multi_head_self_attention(nn.Module):
    def __init__(self):
        super(multi_head_self_attention, self).__init__()
        self.W_Q = nn.Linear(embedding_size, dimension_k * n_heads, #输出维度是64 8个输出头
                             bias=False) 
        self.W_K = nn.Linear(embedding_size, dimension_k * n_heads, bias=False)
        self.W_V = nn.Linear(embedding_size, dimension_v * n_heads, bias=False)
        
        self.linear = nn.Linear(n_heads * dimension_v, embedding_size, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):#torch.Size([2, 11, 512]) torch.Size([2, 11, 11])

        residual, batch_size = input_Q, input_Q.size(0)
        # Q: [batch_size, n_heads, len_q, dimension_k]
        Q = self.W_Q(input_Q).view(batch_size, -1,  #torch.Size([2, 11, 512])
                                   n_heads, dimension_k).transpose(1, 2) #torch.Size([2, 8, 11, 64]) 
        # K: [batch_size, n_heads, len_k, dimension_k] 
        K = self.W_K(input_K).view(batch_size, -1,
                                   n_heads, dimension_k).transpose(1, 2) #torch.Size([2, 8, 11, 64])
        # V: [batch_size, n_heads, len_v(=len_k), dimension_v]
        V = self.W_V(input_V).view(batch_size, -1,
                                   n_heads, dimension_v).transpose(1, 2) #torch.Size([2, 8, 11, 64])

        # attn_mask: [batch_size, token_len, token_len] -> [batch_size, n_heads, token_len, token_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) #torch.Size([2, 1, 11, 11]) torch.Size([2, 8, 11, 11])
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask) 
        context = context.transpose(1, 2).reshape( 
            batch_size, -1, n_heads * dimension_v)

        output = self.linear(context)  # [batch_size, len_q, embedding_size]  #torch.Size([2, 11, 512])
        return nn.LayerNorm(embedding_size).to(device)(output + residual), attn #torch.Size([2, 11, 512]) torch.Size([2, 8, 11, 11])
    
class pos_wise_feed_forward_net(nn.Module): #Position-wise Feed-Forward Neural Network
    def __init__(self):
        super(pos_wise_feed_forward_net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, dimension_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(dimension_hidden, embedding_size, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, token_len, embedding_size]
        """
        residual = inputs
        output = self.fc(inputs)
        # [batch_size, token_len, embedding_size]
        return nn.LayerNorm(embedding_size).to(device)(output + residual)

class EncoderLayer(nn.Module): #encoderlayer contain multi head and pos
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attention = multi_head_self_attention() 
        self.pos_wise_feed_forward_net = pos_wise_feed_forward_net()

    def forward(self, enc_inputs, enc_mask): #torch.Size([2, 11, 512]) torch.Size([2, 11, 11])
        enc_outputs, attn = self.enc_self_attention(enc_inputs, enc_inputs, enc_inputs,
                                               enc_mask)  # torch.Size([2, 11, 512]) torch.Size([2, 11, 11])
        enc_outputs = self.pos_wise_feed_forward_net(enc_outputs)  #torch.Size([2, 11, 512])
        # enc_outputs: [batch_size, sorce_token_len, embedding_size]
        return enc_outputs, attn





class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.enc_embedding = nn.Embedding(source_vocab_size, embedding_size)
        self.pos_embedding = pos_embedding(embedding_size)  
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.enc_embedding(enc_inputs)  # torch.Size([2, 11]) #torch.Size([2, 11, 512])
        enc_outputs = self.pos_embedding(enc_outputs.transpose(0, 1)).transpose(0, 1)#torch.Size([11, 2, 512]) torch.Size([2, 11, 512])
        enc_mask = attn_pad_mask(enc_inputs, enc_inputs) #torch.Size([2, 11])->torch.Size([2, 11,11])
        enc_self_attentions = []  
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 生成输入的每个单词对应的512维向量 输入单词之间的关系
            enc_outputs, multi_head_self_attention = layer(enc_outputs, #torch.Size([2, 11, 512])  Series Network
                                               enc_mask)  # torch.Size([2, 11, 11])
            enc_self_attentions.append(multi_head_self_attention)  # 这个只是为了可视化
        return enc_outputs, enc_self_attentions





class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = multi_head_self_attention()
        self.dec_enc_attn = multi_head_self_attention()
        self.pos_wise_feed_forward_net = pos_wise_feed_forward_net() #在经过线性变化
# layer(dec_outputs, enc_outputs, dec_self_attn_mask,
#                                                              dec_enc_attn_mask) 
    def forward(self, dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, target_token_len, embedding_size] 10
        enc_outputs: [batch_size, sorce_token_len, embedding_size]  11
        dec_self_attn_mask: [batch_size, target_token_len, target_token_len]  10 10
        dec_enc_attn_mask: [batch_size, target_token_len, sorce_token_len]  10 11
        """ ##torch.Size([2, 10, 512]) torch.Size([2, 11, 512]) torch.Size([2, 10, 10]) torch.Size([2, 10, 11])

        dec_outputs, dec_self_attn = self.dec_self_attn(dec_outputs, dec_outputs, dec_outputs,
                                                        dec_self_attn_mask)  
        # dec_outputs: [batch_size, target_token_len, embedding_size], dec_enc_attn: [batch_size, h_heads, target_token_len, sorce_token_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, 
                                                      dec_enc_attn_mask)  #torch.Size([2, 10, 512]) torch.Size([2, 11, 512])
        # [batch_size, target_token_len, embedding_size]
        dec_outputs = self.pos_wise_feed_forward_net(dec_outputs)
        # dec_self_attn, dec_enc_attn这两个是为了可视化的
        return dec_outputs, dec_self_attn, dec_enc_attn
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec_embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.pos_embedding = pos_embedding(embedding_size) #嵌入位置信息
        self.layers = nn.ModuleList([DecoderLayer()for _ in range(n_layers)])  
                    # torch.Size([2, 11]) torch.Size([2, 11, 512]) torch.Size([2, 10])
    def forward(self, enc_inputs, enc_outputs, dec_inputs):

        dec_outputs = self.dec_embedding(dec_inputs)  
        dec_outputs = self.pos_embedding(dec_outputs.transpose(0, 1)).transpose(0, 1).to( device)  

        dec_self_attn_pad_mask = attn_pad_mask(dec_inputs, dec_inputs).to( device)  # [batch_size, target_token_len, target_token_len]  #torch.Size([2, 10, 10])

        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(
            device)  # [batch_size, target_token_len, target_token_len]  #torch.Size([2, 10, 10])

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).to(device)
        dec_enc_attn_mask = attn_pad_mask(
            dec_inputs, enc_inputs) 

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask) #torch.Size([2, 10, 512]) torch.Size([2, 11, 512]) torch.Size([2, 10, 10]) torch.Size([2, 10, 11])
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, target_token_len, embedding_size]
        return dec_outputs, dec_self_attns, dec_enc_attns
    

# 包含编码器解码器 和映射层
class Transformer(nn.Module): 
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.linear = nn.Linear(
            embedding_size, target_vocab_size, bias=False).to(device) #输出最后的17维向量

    def forward(self, enc_inputs, dec_inputs): #torch.Size([2, 11]) torch.Size([2, 10])

        enc_outputs, enc_self_attentions = self.encoder(enc_inputs) #torch.Size([2, 11])

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            enc_inputs, enc_outputs, dec_inputs )   #torch.Size([2, 10])  torch.Size([2, 11])  torch.Size([2, 11, 512])    
        dec_logits = self.linear(dec_outputs) #torch.Size([2, 10, 17])
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attentions, dec_self_attns, dec_enc_attns
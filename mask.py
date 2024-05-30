import torch
import numpy as np

def attn_pad_mask(token_1, token_2): #torch.Size([2, 11])

    batch_size, len_1 = token_1.size()  #2 11 
    batch_size, len_2 = token_2.size()  #2 11

    pad_attn_mask = token_2.data.eq(0).unsqueeze(1) #torch.Size([2, 11])
    return pad_attn_mask.expand(batch_size, len_1, len_2) #2 11 11


def get_attn_subsequence_mask(token): #torch.Size([2, 10]) 解码输入
    """建议打印出来看看是什么的输出（一目了然）
    token: [batch_size, target_token_len]
    """
    attn_shape = [token.size(0), token.size(1), token.size(1)]
    # attn_shape: [batch_size, target_token_len, target_token_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, target_token_len, target_token_len]
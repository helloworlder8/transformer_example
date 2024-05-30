# ====================================================================================================
import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from transformer_model import *
from data_utils import *


model = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3,
                      momentum=0.99)  

# ====================================================================================================
for epoch in range(epochs):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        """
        enc_inputs: [batch_size, sorce_token_len]
        dec_inputs: [batch_size, target_token_len]
        dec_outputs: [batch_size, target_token_len]
        """
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(
            device), dec_inputs.to(device), dec_outputs.to(device) #torch.Size([2, 7])
        # outputs: [batch_size * target_token_len, target_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(
            enc_inputs, dec_inputs) #编码输入 解码输入
        # dec_outputs.view(-1):[batch_size * target_token_len * target_vocab_size]
        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def greedy_decoder(model, enc_input, start_symbol): #torch.Size([1, 11])
    enc_outputs, enc_self_attns = model.encoder(enc_input) #torch.Size([1, 11, 512]) 编码器最终输入值
    # 初始化一个空的tensor: tensor([], size=(1, 0), dtype=torch.int64)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
                              -1)
        dec_outputs, _, _ = model.decoder(enc_input, enc_outputs, dec_input) #torch.Size([1, 1]) torch.Size([1, 11]) torch.Size([1, 11, 512])
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == target_vocab["E"]:
            terminal = True
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict


# ==========================================================================================
# 预测阶段
# 测试集
sentences = [
    # enc_input                dec_input           dec_output
    ['我 有 一 个 女 朋 友 , 你 呢 P', '', '']
]

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
test_loader = Data.DataLoader(
    MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
enc_inputs, _, _ = next(iter(test_loader))

print()
print("="*30)
print("利用训练好的Transformer模型将中文句子'我 有 零 个 女 朋 友' 翻译成英文句子: ")
for i in range(len(enc_inputs)):
    greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(
        1, -1).to(device), start_symbol=target_vocab["S"])
    print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
    print([source_idx2word[t.item()] for t in enc_inputs[i]], '->',
          [target_idx2word[n.item()] for n in greedy_dec_predict.squeeze()])


# 逻辑应该没问题，数据集太小了的原因，网络拟合能力比较大



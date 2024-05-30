
import torch
#device = 'cpu'
device = 'cuda'

# transformer epochs
epochs = 100

embedding_size = 512 
dimension_hidden = 2048
dimension_k = dimension_v = 64  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
n_layers = 6  # number of Encoder of Decoder Layer（number of Block）
n_heads = 8  # number of heads in Multi-Head Attention

sentences = [
    # The number of words in Chinese and English does not need to be the same
    # enc_input                dec_input           dec_output
    ['我 有 一 个 好 朋 友 , 你 呢 P', 'S I have a good friend how about you .', 'I have a good friend how about you . E'],
    ['我 有 零 个 女 朋 友 , 你 呢 P', 'S I have zero girl friend how about you .', 'I have zero girl friend how about you . E'],
    ['我 有 一 个 男 朋 友 , 你 呢 P', 'S I have a boy friend how about you .', 'I have a boy friend how about you . E']
]


source_vocab = {'P': 0, '我': 1, '有': 2, '一': 3,
             '个': 4, '好': 5, '朋': 6, '友': 7, '零': 8, '女': 9, '男': 10, ',': 11, '你': 12, '呢':13}
source_vocab_size = len(source_vocab)
source_idx2word = {i: w for i, w in enumerate(source_vocab)}


target_vocab = {'P': 0, 'I': 1, 'have': 2, 'a': 3, 'good': 4,
             'friend': 5, 'zero': 6, 'girl': 7,  'boy': 8, 'S': 9, 'E': 10, '.': 11, ',': 12, 
             'how': 13, 'about':14, 'you':14, 'new':15}
target_vocab_size = len(target_vocab)
target_idx2word = {i: w for i, w in enumerate(target_vocab)}


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
 
        enc_input = [[source_vocab[n] for n in sentences[i][0].split()]] 
        dec_input = [[target_vocab[n] for n in sentences[i][1].split()]] 
        dec_output = [[target_vocab[n] for n in sentences[i][2].split()]]

        #[[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 8, 4, 9, 6, 7, 0], [1, 2, 3, 4, 10, 6, 7, 0]]
        enc_inputs.extend(enc_input) #end of zero
        #[[9, 1, 2, 3, 4, 5, 11], [9, 1, 2, 6, 7, 5, 11], [9, 1, 2, 3, 8, 5, 11]]
        dec_inputs.extend(dec_input) #start with 9
        #[[1, 2, 3, 4, 5, 11, 10], [1, 2, 6, 7, 5, 11, 10], [1, 2, 3, 8, 5, 11, 10]]
        dec_outputs.extend(dec_output) #end of ten

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs) #Convert to tensor format


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
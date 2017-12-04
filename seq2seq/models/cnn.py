import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierCNN(nn.Module):
    def __init__(self, num_vocab, embed_dim, num_class, num_kernel, kernel_sizes, dropout_p):
        super(ClassifierCNN, self).__init__()

        self.embed = nn.Embedding(num_vocab, embed_dim)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_kernel, (k, embed_dim)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout_p)
        self.conv2out = nn.Linear(len(kernel_sizes) * num_kernel, num_class)

    def forward(self, x):
        # print(x.size())
        x = self.embed(x)  # (batch, input_size, embed_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_size, embed_dim)
        # print(x.size())
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch, C_out, Width), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch,C_out), ...]*len(Ks)
        x = torch.cat(x, 1)  # (batch, len(kernal_sizes) * kernal_num)

        x = self.dropout(x)  # N x len(Ks)*Co
        logit = self.conv2out(x)  # batch x class_num
        out = F.sigmoid(logit)  # out介於[0-1]，表示prob。class_num > 1，改為softmax
        return out

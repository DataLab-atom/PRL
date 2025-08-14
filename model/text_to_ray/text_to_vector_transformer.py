import math
import torch
import torch.nn as nn

# 3. 定义位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        d_model: 嵌入维度
        dropout: dropout 概率
        max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码矩阵，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 按公式计算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# 4. 定义 Transformer 模型
class TextToVectorTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, hidden_dim, num_encoder_layers, output_dim, max_len=5000, dropout=0.1):
        """
        vocab_size: 词汇表大小
        embed_dim: 词嵌入维度
        nhead: Transformer 多头注意力头数
        hidden_dim: Transformer 中 feed-forward 层的维度
        num_encoder_layers: Transformer Encoder 层数
        output_dim: 输出向量的维度
        max_len: 位置编码的最大长度
        dropout: dropout 概率
        """
        super(TextToVectorTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # 这里采用简单的池化（例如均值池化）作为句子表示
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        # 1. 嵌入层
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        # 2. 加入位置编码
        x = self.pos_encoder(x)  # [batch_size, seq_len, embed_dim]
        # 3. Transformer Encoder
        transformer_out = self.transformer_encoder(x)  # [batch_size, seq_len, embed_dim]
        # 4. 池化：这里取均值，忽略padding部分（可以根据需要改进）
        # 计算每个样本的实际长度（非0元素数量）
        mask = (x.sum(dim=-1) != 0).float()  # [batch_size, seq_len]
        lengths = mask.sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        pooled = transformer_out.sum(dim=1) / lengths  # 均值池化
        # 5. 全连接层映射到目标向量
        output = self.fc(pooled)  # [batch_size, output_dim]
        return output
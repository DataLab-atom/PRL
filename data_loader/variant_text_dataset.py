import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from collections import Counter
import math
import random


# ---------------------------
# 定义数据集：每个样本只包含文本和类别标签
class VariantTextDataset(Dataset):
    def __init__(self, variants_dict, vocab, tokenize_func, max_len=10):
        """
        variants_dict: dict, key 为类别名称，value 为文本变体列表
        vocab: 词汇表
        tokenize_func: 分词函数
        max_len: 最大 token 数
        """
        self.samples = []
        for cat, texts in variants_dict.items():
            for text in texts:
                self.samples.append((text, cat))
        self.vocab = vocab
        self.tokenize = tokenize_func
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, cat = self.samples[idx]
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, 0) for token in tokens]
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [0]*(self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        return token_tensor, cat
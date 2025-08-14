import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from collections import Counter
import math
import random

# ---------------------------
# 自定义 DataLoader：按类别分组，每次返回同一类别的所有文本和对应的目标向量集合
class VariantDataLoader:
    def __init__(self, dataset, category_target_dict):
        self.samples_by_cat = {}
        for sample in dataset:
            token_tensor, cat = sample
            if cat not in self.samples_by_cat:
                self.samples_by_cat[cat] = []
            self.samples_by_cat[cat].append(token_tensor)
        self.category_target_dict = category_target_dict
        self.categories = list(self.samples_by_cat.keys())

    def __iter__(self):
        random.shuffle(self.categories)
        for cat in self.categories:
            texts = torch.stack(self.samples_by_cat[cat], dim=0)
            target_vectors = torch.tensor(self.category_target_dict[cat], dtype=torch.float)
            yield cat, texts, target_vectors

    def __len__(self):
        return len(self.categories)
import torch
from torch.utils.data import Dataset

class TextVectorDataset(Dataset):
    def __init__(self, data_pairs, vocab, tokenize_func, max_len=20):
        self.data_pairs = data_pairs
        self.vocab = vocab
        self.tokenize = tokenize_func
        self.max_len = max_len

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        text, vector = self.data_pairs[idx]
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, 0) for token in tokens]
        # 截断或填充，保证长度一致
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [0] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        vector_tensor = torch.tensor(vector, dtype=torch.float)
        return token_tensor, vector_tensor
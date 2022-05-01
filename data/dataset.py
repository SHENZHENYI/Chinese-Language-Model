import torch
from torch.utils.data import Dataset
from data.vocab import Vocabulary

class CorpusData(Dataset):
    def __init__(self, corpus_path, vocab=None, min_freq=3):
        with open(corpus_path, 'r', encoding='utf8') as f:
            self.corpus = f.readlines()
            self.corpus = [line.replace('\n', '') for line in self.corpus if len(line.replace('\n', ''))>1]
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(min_freq=3)
            self.vocab.build(self.corpus)
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        sent = self.corpus[idx]
        x = sent[:-1]
        y = sent[1:]
        return torch.tensor(self.vocab.numericalize(x)).long(), \
                torch.tensor(self.vocab.numericalize(y)).long()

    def get_vocab(self):
        return self.vocab
        
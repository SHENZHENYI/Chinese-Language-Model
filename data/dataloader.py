from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def get_dataloader(dataset, batch_size, num_workers, shuffle):
    pad_idx = dataset.vocab.char2idx['<PAD>']
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle, collate_fn = MyCollate(pad_idx=pad_idx))
    return loader

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        x, y = zip(*batch)
        # get len
        x_lens = list(map(len, x))
        y_lens = list(map(len, y))

        # do padding
        x_pad = pad_sequence(x, batch_first=False, padding_value=self.pad_idx)
        y_pad = pad_sequence(y, batch_first=False, padding_value=self.pad_idx)

        return x_pad, y_pad, x_lens, y_lens


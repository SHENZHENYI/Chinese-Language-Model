from torch import nn 

class RNNModel(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=num_layers)
        self.fc = nn.Linear(hid_dim, input_dim)
    
    def forward(self, input, input_len, hidden=None):
        embedded = self.embedding(input)
        # pad
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_len, enforce_sorted=False)
        # gru could operate on PackedSequence
        packed_outputs, hidden = self.rnn(packed_embedded, hidden)
        # unpack
        padded_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        padded_outputs = self.fc(padded_outputs)
        return padded_outputs, hidden


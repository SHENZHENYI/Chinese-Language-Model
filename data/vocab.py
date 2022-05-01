class Vocabulary:
    def __init__(self, min_freq):
        self.min_freq = min_freq
        self.idx2char = {0:'<PAD>', 1:'<UNK>'}
        self.char2idx = {v:k for k, v in self.idx2char.items()}

    def __len__(self):
        return len(self.char2idx)

    def tokenize(self, sent: str) -> list:
        '''string text to a list of tokens'''
        tokens = list(sent)
        return tokens

    def numericalize(self, sent: str) -> list:
        '''token to idx'''
        idxs = []
        tokens = self.tokenize(sent)
        for token in tokens:
            if token in self.char2idx:
                idxs.append(self.char2idx[token])
            else:
                idxs.append(self.char2idx['<UNK>'])
        return idxs

    def build(self, sents: list) -> None:
        '''build the vocab'''
        freqs = dict()
        idx = len(self.idx2char)

        for sent in sents:
            for char_ in sent:
                if char_ not in freqs:
                    freqs[char_] = 0
                freqs[char_] += 1
        
        # remove low freq chars
        freqs = {char_: value for char_, value in freqs.items() if value >= self.min_freq}

        # build
        for char_ in sorted(freqs.keys()):
            self.char2idx[char_] = idx
            self.idx2char[idx] = char_ 
            idx += 1
        
        print(f'Built the vocab. The total vocab len is {len(self)}')
    

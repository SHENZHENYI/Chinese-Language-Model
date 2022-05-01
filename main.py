import random
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F

from data.dataset import CorpusData
from data.dataloader import get_dataloader
from model.rnn import RNNModel

def train(n_epochs, model, dataloader, loss_fn, optimizer, device, vocab, model_path, clip, pred_seq_len=200):
    '''Train

    This function trains the model by wrapping 'train_one_epoch'

    Args:
        n_epochs: number of epochs to be trained
        model: the model
        raw_data: one big string that holds all training data
        loss_fn: loss function
        optimizer: optimizer

    Returns:
        no returns. the model will be updated.
    '''
    for epoch in range(n_epochs):
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device, model_path, clip)
        print(f'epoch: {epoch}, loss: {loss}')
        predict(model, device, vocab, seq_len=pred_seq_len, start_char='又')

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, model_path, clip):
    '''Training process in one epoch

    Args:
        model: the neural network
        raw_data: one big string that holds all training data
        loss_fn: loss function
        optimizer: optimizer
        batchsize: the batch size in the training process
        seq_len: the seq length in the training process

    Returns:
        the training loss after training the epoch.
        model is not returned, but will be updated.
    '''
    model.train()
    loss_meter = []
    with tqdm(total=len(dataloader)) as prog:
        for x, y, x_len, y_len in dataloader:
            optimizer.zero_grad()
            pred, _ = model(x.to(device), x_len)
            loss = 0
            for i in range(pred.shape[1]):
                loss += loss_fn(pred[:,i], y[:,i].to(device))/pred.shape[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            prog.update(1)
            loss_meter.append(loss.item())
            save_model(model, model_path)
            avg_loss = sum(loss_meter)/len(loss_meter)
    return sum(loss_meter)/len(loss_meter)


def predict(model, device, vocab, seq_len=200, start_char='明'):
    '''Predict a sequence with a length of 200 when fed with a random char

    Args:
        model: the rnn
        raw_data: the training data, we will only sample one char from it, nothing elses
        seq_len: seq len of the predicted linen

    Returns:
        the result predicted string
    '''
    model.eval()
    input = vocab.char2idx[start_char]
    #input = F.one_hot(encode_str_data(input_seq), num_classes=95).to(torch.float32)
    out_str = []
    hidden = None
    for i in range(seq_len):
        input = torch.tensor(input).long()
        output, hidden = model(input.view(1,-1).to(device), [1], hidden)
        probs = F.softmax(output.view(-1)).cpu().tolist() # to prob
        sample = random.choices(list(vocab.char2idx.keys()), weights=probs)[0] # sample with the prob
        input = vocab.char2idx[sample]
        out_str.append(sample)
        if sample == '。': # training data stops at a full stop
            break
    print('*'*60)
    print(f'The predicted string with a starting seed of {start_char}')
    print('*'*60)
    print(start_char+''.join(out_str))
    print('*'*60)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model, save_path, device):
    model.load_state_dict(torch.load(save_path, map_location=device))

def main():
    # configs
    LEARNING_RATE = 0.002
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    CLIP = 1.
    EMB_DIM = 256
    HID_DIM = 128
    NUM_LAYERS = 2
    N_EPOCHS = 10
    #########
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus_path = './corpus/songci.txt'
    model_path = './ckpts/rnn.pt'
    train_set = CorpusData(corpus_path)
    vocab = train_set.get_vocab()
    model = RNNModel(input_dim=len(vocab), emb_dim=EMB_DIM, hid_dim=HID_DIM, num_layers=NUM_LAYERS).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = get_dataloader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    
    train(N_EPOCHS, model, train_loader, loss_fn, optimizer, device, vocab, model_path, CLIP, pred_seq_len=200)


if __name__ == '__main__':
    main()


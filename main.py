import dataset
from model import CharRNN, CharLSTM
from generate import *

import argparse
import time
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader


parser = argparse.ArgumentParser(description='Shakespeare_LM')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay (L2 penalty)')
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--hidden_size', default=512, type=int)
parser.add_argument('--generate', default=True, type=bool)
parser.add_argument('--gen_len', default=300, type=int)

args = parser.parse_args()

def set_env(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpuid)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def loss_plot(filename, train_loss_history, val_loss_history, best_epoch):
    plt.figure(figsize=(12, 6))

    # Loss Plot (Training and Validation)
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'b-', label='Train Loss', linewidth=2)
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, 'r-', label='Validation Loss', linewidth=2)

    # Highlight the best epoch
    plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch')

    plt.title('Train & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Setting the x-axis limits
    plt.xlim(1, len(train_loss_history))  # This will set the x-axis from 1 to the number of epochs

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def model_comparison(filename, rnn_test_loss_history, lstm_custom_test_loss_history):
    plt.figure(figsize=(12, 6))

    # Testing Loss Comparison
    plt.plot(range(1, len(rnn_test_loss_history) + 1), rnn_test_loss_history, 'r-', label='Vanilla RNN validation Loss', linewidth=2)  
    plt.plot(range(1, len(lstm_custom_test_loss_history) + 1), lstm_custom_test_loss_history, 'b-', label='LSTM validation Loss', linewidth=2)  
    plt.title('validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def train(model, trn_loader, device, criterion, optimizer, epoch, args):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    model.train()
    total_loss = 0
    total = 0
    start_time= time.time()

    for batch_idx, (inputs, targets) in enumerate(trn_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0))

        if isinstance(hidden, tuple):
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:
            hidden = hidden.to(device)

        outputs, hidden = model(inputs, hidden)
        hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()

        # (batch_size * seq_len, vocab_size) / (batch_size * seq_len,)
        loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total += targets.size(0)

        elapsed_time = time.time() - start_time  # 현재까지의 경과 시간 계산
        hours, rem = divmod(elapsed_time, 3600)  # 시간 및 나머지 계산
        minutes, seconds = divmod(rem, 60)  # 분 및 초 계산
        time_str = "{:0>2}:{:0>2}:{:02.0f}".format(int(hours), int(minutes), seconds)
        
        # Monitoring
        if (batch_idx + 1) % 100 == 0:
            sys.stdout.write('\rEpoch [%3d/%3d] | Iter[%3d/%3d] | Elapsed Time %s | loss: %.4f' 
                                % (epoch, args.num_epochs, batch_idx + 1, len(trn_loader), time_str, loss.item()))
            sys.stdout.flush()

    trn_loss = total_loss / total

    return trn_loss

def validate(model, val_loader, device, criterion, epoch, args):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for valing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """
    model.eval()
    total_loss = 0
    total = 0
    start_time= time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))

            if isinstance(hidden, tuple):
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = hidden.to(device)

            outputs, hidden = model(inputs, hidden)
            hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()

            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            total_loss += loss.item() * inputs.size(0)
            total += targets.size(0)

    val_loss = total_loss / total

    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct dataloader / (size) train:val = 0.8:0.2
    full_dataset = dataset.Shakespeare('./dataset/shakespeare_train.txt')
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    split_point = int(np.floor(0.2 * len(indices)))
    val_indices = indices[:split_point]
    train_indices = indices[split_point:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(full_dataset, 
                              batch_size = args.batch_size,
                              num_workers = 4,
                              sampler = train_sampler)
    
    val_loader = DataLoader(full_dataset, 
                            batch_size = args.batch_size,
                            num_workers = 4,
                            sampler = val_sampler)

    # Model init
    vocab_size = len(full_dataset.index_to_char)
    # print(vocab_size)
    model_rnn = CharRNN(vocab_size, args.embedding_dim, args.hidden_size, vocab_size, args.n_layers).to(device)
    model_lstm = CharLSTM(vocab_size, args.embedding_dim, args.hidden_size, vocab_size, args.n_layers).to(device)

    # optim & Cost function init
    criterion = torch.nn.CrossEntropyLoss()
    rnn_optimizer = torch.optim.AdamW(model_rnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lstm_optimizer = torch.optim.AdamW(model_lstm.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    # training & validation phase
    print('============Vanilla RNN training & validation Start============')
    rnn_train_loss_history, rnn_val_loss_history = [], []
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        trn_loss = train(model_rnn, train_loader, device, criterion, rnn_optimizer, epoch+1, args)
        val_loss = validate(model_rnn, val_loader, device, criterion, epoch, args)
        print("\n(Epoch %d) Train loss : %.3f | Val loss : %.3f \n" % (epoch+1, trn_loss, val_loss))

        rnn_train_loss_history.append(trn_loss)
        rnn_val_loss_history.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            rnn_best_epoch = epoch+1
            best_rnn_state_dict = model_rnn.state_dict()
    torch.save(best_rnn_state_dict, f'./save_model/best_model_rnn_epoch{rnn_best_epoch}_layers_{args.n_layers}.pth')        
    loss_plot(f'./plot/rnn_loss_plot_layers_{args.n_layers}', rnn_train_loss_history, rnn_val_loss_history, rnn_best_epoch)

    print('============LSTM training & validation Start============')
    lstm_train_loss_history, lstm_val_loss_history = [], []
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        trn_loss = train(model_lstm, train_loader, device, criterion, lstm_optimizer, epoch+1, args)
        val_loss = validate(model_lstm, val_loader, device, criterion, epoch, args)
        print("\n(Epoch %d) Train loss : %.3f | Val loss : %.3f \n" % (epoch+1, trn_loss, val_loss))
        
        lstm_train_loss_history.append(trn_loss)
        lstm_val_loss_history.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            lstm_best_epoch = epoch+1
            best_lstm_state_dict = model_lstm.state_dict()
    torch.save(best_lstm_state_dict, f'./save_model/best_model_lstm_epoch{lstm_best_epoch}_layers_{args.n_layers}.pth')
    loss_plot(f'./plot/lstm_loss_plot_layers_{args.n_layers}', lstm_train_loss_history, lstm_val_loss_history, lstm_best_epoch)

    # save plot
    model_comparison(f'./plot/model_comparison_plot_layers_{args.n_layers}',rnn_val_loss_history, lstm_val_loss_history)

    if args.generate:
        model_rnn.load_state_dict(torch.load(f'./save_model/best_model_rnn_epoch{rnn_best_epoch}_layers_{args.n_layers}.pth'))
        best_rnn = model_rnn
        model_lstm.load_state_dict(torch.load(f'./save_model/best_model_lstm_epoch{lstm_best_epoch}_layers_{args.n_layers}.pth'))
        best_lstm = model_lstm

        seed_characters_list = ['QUEEN: So, lets end this', 'Lord: Kill him!', 'First Citizen: Before we proceed', 'Romeo: I love you', 'All:']
        temperatures = [0.1, 0.3, 0.5, 0.8, 1, 1.5, 2]

        rnn_texts = []
        lstm_texts = []
        for temperature in temperatures:
            print(f'================= temperature : {temperature} =================')
            for seed_characters in seed_characters_list:
                rnn_gen_text = generate(best_rnn, seed_characters, temperature, device, full_dataset, len = args.gen_len)
                lstm_gen_text = generate(best_lstm, seed_characters, temperature, device, full_dataset, len = args.gen_len)
                rnn_texts.append(f"Temperature {temperature}, Seed: {seed_characters}\n{rnn_gen_text}")
                lstm_texts.append(f"Temperature {temperature}, Seed: {seed_characters}\n{lstm_gen_text}")

                print(f'################ Temperature {temperature}, Seed: {seed_characters} ################')
                print('==========Generated text by Vanilla RNN==========')
                print(rnn_gen_text)
                print('==========Generated text by LSTM==========')
                print(lstm_gen_text)

        save_generated_text(f'generated_text/generated_texts_rnn_{args.n_layers}_{args.num_epochs}.txt', rnn_texts)
        save_generated_text(f'generated_text/generated_texts_lstm_{args.n_layers}_{args.num_epochs}.txt', lstm_texts)

if __name__ == '__main__':
    # set seed
    set_env(args)

    # training
    main()
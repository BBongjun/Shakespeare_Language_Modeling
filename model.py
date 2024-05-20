import torch.nn as nn
import torch.nn.functional as F
import torch

class CharRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, n_layers=1, dropout_rate = 0.2):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size = input_size

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)  
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)

    def forward(self, input, hidden):
        x = self.embedding(input)
        x, hidden = self.rnn(x, hidden)
        # x = self.dropout(x)
        x = x.contiguous().view(-1, self.hidden_size)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        output = self.fc2(x)
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, n_layers=1, dropout_rate=0.2):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size = input_size

        self.embedding = nn.Embedding(input_size, embedding_dim)        
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)  
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)

    def forward(self, input, hidden):
        x = self.embedding(input)
        x, hidden = self.lstm(x, hidden)
        # x = self.dropout(x)
        x = x.contiguous().view(-1, self.hidden_size)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        output = self.fc2(x)
        return output, hidden

    def init_hidden(self, batch_size):
        # (hidden state , cell state)
        initial_hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                          torch.zeros(self.n_layers, batch_size, self.hidden_size))
        
        return initial_hidden
import torch
import torch.nn as nn
from fastgc.layers.recurrent import RNNCell
from fastgc.layers.recurrent import RNNModule
from fastgc.layers.recurrent import LSTMCell
from fastgc.layers.linear import Linear
from fastgc.model.penet import PeGradNet


class RNN(PeGradNet):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1,
                 train_alg='batch', bias=True):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
        #                   nonlinearity='tanh')
        self.rnn = RNNModule(input_size, hidden_size)
        self.output_layer = Linear(hidden_size, num_classes)
        self.train_alg = train_alg

        self.layers = [self.rnn, self.output_layer]

    def forward(self, x):
        # x = x.squeeze(1).permute(1, 0, 2)  # seq_len x batch_size x input_size
        batch_size = x.shape[0]
        x = x.reshape(batch_size, x.shape[2], -1)
        x = x.permute(1, 0, 2)
        seq_len = x.shape[0]        
        num_directions = 2 if self.rnn.bidirectional else 1
        S = self.rnn.num_layers * num_directions

        h0 = torch.zeros(S, batch_size, self.hidden_size, device=x.device)
        out, hn = self.rnn(x, h0)

        # if self.train():
        #     out.retain_grad()

        logits = self.output_layer(out[-1])

        return logits


class SimpleLSTM(PeGradNet):
    def __init__(self, input_size, hidden_size, output_size, train_alg='batch'):
        super(SimpleLSTM, self).__init__()
        
        self.lstm = LSTMCell(input_size, hidden_size)
        self.fc = Linear(hidden_size, output_size)
        self.train_alg = train_alg

    def forward(self, x, init_states=None):
        # x = x.squeeze(1)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, x.shape[2], -1)
        seq_size = x.shape[1]
        hidden_size = self.lstm.hidden_size

        self.lstm.reset_pgrad()

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, hidden_size, device=x.device), 
                        torch.zeros(batch_size, hidden_size, device=x.device))
        else:
            h_t, c_t = init_states
         
        for t in range(seq_size):
            x_t = x[:, t, :]
            h_t, c_t = self.lstm(x_t, h_t, c_t)

        logits = self.fc(h_t)

        return logits

    def pe_grad_norm(self, loss, batch_size, device):
        grad_norm = torch.zeros(batch_size, device=device)

        pre_acts = self.lstm.pre_activation
        pre_acts.append(self.fc.pre_activation)
        Z_grad = torch.autograd.grad(loss, pre_acts, retain_graph=True)

        grad_norm.add_(self.lstm.pe_grad_sqnorm(Z_grad[:-1]))
        grad_norm.add_(self.fc.pe_grad_sqnorm(Z_grad[-1]))
            
        grad_norm.sqrt_()

        return grad_norm


class SimpleRNN(PeGradNet):
    def __init__(self, input_size, hidden_size, num_classes, train_alg='batch'):
        super(type(self), self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = num_classes
        self.train_alg = train_alg

        self.rnn = RNNCell(input_size, hidden_size)
        self.fc = Linear(self.hidden_size, self.output_size)


    def forward(self, x):
        x = x.squeeze(1).permute(1, 0, 2)  # seq_len x batch_size x input_size

        self.rnn.reset_pgrad()

        hx = torch.zeros(x.shape[1], self.hidden_size, device=x.device)
        
        for t in range(x.shape[0]):
            hx = self.rnn(x[t], hx)

        logits = self.fc(hx)

        return logits

    def per_example_gradient(self, loss):
        grads = []

        pre_acts = self.rnn.pre_activation
        pre_acts.append(self.fc.pre_activation)
        
        Z_grad = torch.autograd.grad(loss, pre_acts, retain_graph=True)

        grads.extend(self.rnn.per_example_gradient(Z_grad[:-1]))
        grads.extend(self.fc.per_example_gradient(Z_grad[-1]))

        return grads

    def pe_grad_norm(self, loss, batch_size, device):
        grad_norm = torch.zeros(batch_size, device=device, requires_grad=False)
        
        pre_acts = self.rnn.pre_activation
        pre_acts.append(self.fc.pre_activation)

        Z_grad = torch.autograd.grad(loss, pre_acts, retain_graph=True)
        grad_norm.add_(self.rnn.pe_grad_sqnorm(Z_grad[:-1]))
        grad_norm.add_(self.fc.pe_grad_sqnorm(Z_grad[-1]))

        grad_norm.sqrt_()

        return grad_norm

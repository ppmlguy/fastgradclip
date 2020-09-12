import torch
import torch.nn as nn
from pegrad.layers.linear import Linear
from pegrad.layers.transformer import PositionalEncoding
from pegrad.layers.transformer import TransformerEncoder
from pegrad.layers.transformer import TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(self, n_token, n_classes, d_model=512, n_layers=2,
                 n_head=8, n_hidden=2048, dropout=0.1, max_seq_len=512,
                 embeddings=None, train_alg='batch'):
        super(TransformerModel, self).__init__()

        self.train_alg = train_alg
        self.d_model = d_model
        self.n_head = n_head

        if embeddings is None:            
            self.token_embedding = nn.Embedding(n_token, d_model)
        else:
            self.token_embedding = nn.Embedding.from_pretrained(embeddings)
            self.token_embedding.weight.requires_grad = False

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)        
        encoder_layers = TransformerEncoderLayer(d_model, n_head, n_hidden, dropout)
        # encoder_norm = nn.LayerNorm(d_model)
        encoder_norm = None
        self.encoder = TransformerEncoder(encoder_layers, n_layers, encoder_norm)
        self.fc= Linear(d_model, n_classes)

    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # positions = torch.arange(len(x), device=x.device).unsqueeze(-1)

        x = x.transpose(0, 1)
        # [sentence length, batch_size]
        x = self.token_embedding(x)
        # [sentence length, batch_size, embedding dim]
        x = self.pos_encoder(x)
        # x = x + self.pos_encoder(positions).expand_as(x)
        
        # [sentence length, batch_size, embedding dim]
        output = self.encoder(x)
        # [sentence length, batch_size, embedding dim]
        avg_out = output.transpose(0, 1).mean(dim=1)
        # [batch_size, embedding dim]
        preact = self.fc(avg_out)
        
        # [batch_size, num_classes]
        # return F.log_softmax(output, dim=-1)
        return preact

    def per_example_gradient(self, loss):
        grads = []
        pre_acts = []

        pre_acts.extend(self.encoder.collect_preactivations())
        pre_acts.append(self.fc.pre_activation)

        pre_acts = [m.pre_activ for m in modules]
        Z_grad = torch.autograd.grad(loss, pre_acts, retain_graph=True)
        for m, zgrad in zip(modules, Z_grad):
            m.save_grad(zgrad)
        # loss.backward(retain_graph=True)        

        # TransformerEncoder
        grads.extend(self.encoder.per_example_gradient())

        # fully connected layer
        grads.extend(self.fc.per_example_gradient())

        return grads

    def pe_grad_norm(self, loss, batch_size, device):
        grad_norm = torch.zeros(batch_size, device=device)

        pre_acts = []
        pre_acts.extend(self.encoder.collect_preactivations())
        pre_acts.append(self.fc.pre_activation)

        Z_grad = torch.autograd.grad(loss, pre_acts, retain_graph=True)

        grad_norm.add_(self.encoder.pe_grad_sqnorm(Z_grad[:-1]))
        grad_norm.add_(self.fc.pe_grad_sqnorm(Z_grad[-1]))        
        grad_norm.sqrt_()

        return grad_norm



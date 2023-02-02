import torch 
from torch import nn 
import torch.nn.functional as F
import panphon2


class LSTM_Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, device, feature_size=24, bidirectional=True, use_attn=True):
        super().__init__()

        self.encoder = torch.nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # This seems to bound the embedding norms at start of training
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param) 

        if bidirectional:
            self.out_dim = 2 * hidden_size
        else:
            self.out_dim = hidden_size
            
        self.use_attn = use_attn
        if self.use_attn:
            self.attn = Attention(self.out_dim)

        self.to(device)
        self.device = device

    def forward(self, x):
        x_pad = torch.nn.utils.rnn.pad_sequence(
                    [torch.Tensor(x_0) for x_0 in x],
                    batch_first=True, padding_value=-1.0,
                ).to(self.device)

        output, (_, _) = self.encoder(x_pad)

        if self.use_attn:
            output = self.attn(output) # attention on hidden states of encoders
        else: 
            output = output[:, -1, :]

        return output

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        u = self.attn(x)
        u = torch.tanh(u)
        alphas = F.softmax(self.context(u), dim=1)
        context = (alphas * x).sum(dim=1)
        return context



        

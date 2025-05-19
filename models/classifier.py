import torch
import torch.nn as nn

class ResClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=76, extract=False, dropout_p=0.5):
        super(
            ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),)
        self.dropout = nn.Sequential(
            nn.Dropout(p=dropout_p)
            )

        self.fc11 = nn.Linear(hidden_dim, out_dim)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, x, training=True):
        _aux = {}
        _return_aux = False
        fc1_emb = self.fc1(x)
        fc1_emb = self.dropout(fc1_emb)

        logit = self.fc11(fc1_emb)
        return logit
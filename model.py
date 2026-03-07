import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, img_channels, num_classes, hidden_size=256):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),

            nn.Dropout2d(0.25),
        )

        self.map_to_seq = nn.Linear(256 * 4, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w)
        conv = conv.permute(0, 2, 1)
        seq = self.map_to_seq(conv)
        seq = self.dropout(seq)
        out, _ = self.rnn(seq)
        out = self.fc(out)
        return F.log_softmax(out, dim=2)
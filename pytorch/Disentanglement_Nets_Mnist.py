import torch.nn as nn
import torch


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


# Encoders S,Z
# Three 5x5 convolotional layers
# with stride 2
# and a dense S/Z dimension layer.
# all with ReLU non-linearities
class EncoderNet(nn.Module):
    def __init__(self, ngpu):
        super(EncoderNet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(1, 6, 5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 12, 5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 24, 5, stride=1),
            nn.ReLU(inplace=True),
            View(-1, 24),
            nn.Linear(24, 16),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


# Decoder
# Mirroring network to the encoders:
# dense layer and three convolutional
# network with upsampling

class DecoderNet(nn.Module):
    def __init__(self, ngpu):
        super(DecoderNet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            View(-1, 1, 32),
            nn.BatchNorm1d(1),
            # View(-1, 32),
            nn.Linear(32, 48),
            # nn.BatchNorm2d(1),
            nn.BatchNorm1d(1),
            View(-1, 48,1,1),
            nn.ReLU(True),
            nn.ConvTranspose2d(48, 12, 5, stride=1),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 6, 5, stride=2),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, 4, stride=2),
            View(-1,784),
            nn.Tanh()
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# (Adversarial) Classifier
# 3 dense layers x 256 hidden units,
# Batch Normalization, ReLU and
# a softmax for the output
class ClassifierNet(nn.Module):
    def __init__(self, ngpu):
        super(ClassifierNet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
            # nn.Softmax(1)
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


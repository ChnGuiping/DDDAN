import math
from torch import nn
import warnings
import torch
import torch.nn.functional as F


class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, frequency, eps, mode='sigmoid'):
        super(Laplace_fast, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.fre = frequency

        # # WCK, then make eps=0, mode='vanilla'
        # self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)
        # self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)
        # self.time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))

        # SWK and EWK
        self.a_ = nn.Parameter(torch.linspace(0, self.out_channels, self.out_channels)).view(-1, 1)
        self.b_ = nn.Parameter(torch.linspace(0, self.out_channels, self.out_channels)).view(-1, 1)
        # self.time_disc = torch.linspace(0, self.kernel_size - 1, steps=int(self.kernel_size))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))
        self.time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))

    def Laplace(self, p):
        # w = 2 * torch.pi * self.fre
        w = torch.pi * self.fre
        A = 0.08
        q = torch.tensor(1 - pow(0.03, 2))

        if self.mode == 'vanilla':
            # return (1 / math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))) * (torch.sin(w * (p - 0.1)))
            return A * torch.exp((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))) * (-torch.sin(w * (p - 0.1)))

        if self.mode == 'sigmoid':
            return (1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (torch.sin(w * (p - 0.1)))
            # return A * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (torch.sin(w * (p - 0.1)))

    def forward(self):
        # p1 = (self.time_disc - self.b_) / (self.a_ + self.eps)
        # return self.Laplace(p1).view(self.out_channels, 1, self.kernel_size)
        p1 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)
        p2 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)
        Laplace_left = self.Laplace(p1)
        Laplace_right = self.Laplace(p2)
        return torch.cat([Laplace_left, Laplace_right], dim=1).view(self.out_channels, 1, self.kernel_size)




class WCNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, init_weights=True):
        super(WCNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=64, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            )

        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.AdaptiveMaxPool1d(4)
            )

        self.gaplayer = nn.AdaptiveAvgPool1d(1)

        self.output_dim = 256*1

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gaplayer(x)
        x = torch.flatten(x, 1)

        return x


    def output_num(self):
        return self.output_dim



# For the same sampling frequency
class WCNN_EWK(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, init_weights=True):
        super(WCNN_EWK, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.eps = nn.Parameter(torch.empty(1).uniform_(0, 0.5), requires_grad=True)
        # self.eps = nn.Parameter(torch.randint(1, 5, (1,), dtype=torch.float32) / 10, requires_grad=True)
        # self.eps = 64000

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=64, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            )

        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.AdaptiveMaxPool1d(4)
            )

        self.gaplayer = nn.AdaptiveAvgPool1d(1)


        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for name, can in self.named_children():
            if name == 'layer1':
                for m in can.modules():
                    if isinstance(m, nn.Conv1d):
                        if m.kernel_size == (64,):
                            m.weight.data = Laplace_fast(out_channels=64, kernel_size=64, eps=self.eps, frequency=200000/3, mode='sigmoid').forward()


        self.output_dim = 256*1


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gaplayer(x)
        x = torch.flatten(x, 1)

        return x


    def output_num(self):
        return self.output_dim




# For different sampling frequencies
class WCNN_EWK1(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, init_weights=True):
        super(WCNN_EWK1, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.eps = args.eps
        # self.eps = nn.Parameter(torch.empty(1).uniform_(0, 0.5), requires_grad=True)

        # two conv layers
        self.layer1_1 = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=64, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            )

        self.layer1_2 = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=64, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            )

        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            )

        self.gaplayer = nn.AdaptiveAvgPool1d(1)


        self.bottleneck_layer = nn.Sequential(nn.Linear(256, 256),
                                              nn.LeakyReLU(inplace=True), nn.Dropout())
        self.classifier_layer = nn.Linear(256, 4)

        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for name, can in self.named_children():
            if name == 'layer1_1':
                for m in can.modules():
                    if isinstance(m, nn.Conv1d):
                        if m.kernel_size == (64,):
                            m.weight.data = Laplace_fast(out_channels=64, kernel_size=64, eps=args.eps, frequency=args.src_frequery, mode=args.mode).forward()      # args.src_frequery
                            nn.init.constant_(m.bias.data, 0.0)
            elif name == 'layer1_2':
                for m in can.modules():
                    if isinstance(m, nn.Conv1d):
                        if m.kernel_size == (64,):
                            m.weight.data = Laplace_fast(out_channels=64, kernel_size=64, eps=args.eps, frequency=args.tar_frequery, mode=args.mode).forward()      # args.tar_frequery
                            nn.init.constant_(m.bias.data, 0.0)

        self.output_dim = 256*1

    def forward(self, x1, x2):
        x1 = self.layer1_1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.gaplayer(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.bottleneck_layer(x1)
        x1 = self.classifier_layer(x1)

        x2 = self.layer1_2(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = self.gaplayer(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.bottleneck_layer(x2)
        x2 = self.classifier_layer(x2)

        return x1, x2

    def scr_predict(self, x1):
        x1 = self.layer1_1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.gaplayer(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.bottleneck_layer(x1)
        x1 = self.classifier_layer(x1)

        return x1

    def tar_predict(self, x2):
        x2 = self.layer1_2(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = self.gaplayer(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.bottleneck_layer(x2)
        x2 = self.classifier_layer(x2)

        return x2
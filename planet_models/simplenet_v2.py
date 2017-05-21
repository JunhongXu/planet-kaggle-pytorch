import torch.nn as nn
import math


class Block(nn.Module):
    def __init__(self, inpt_channel, output_channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inpt_channel, kernel_size=1, out_channels=output_channel//2, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(output_channel//2)
        self.elu1 = nn.ELU()

        self.conv2 = nn.Conv2d(in_channels=output_channel//2, out_channels=output_channel//2, bias=False,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel//2)
        self.elu2 = nn.ELU()

        self.conv3 = nn.Conv2d(in_channels=output_channel//2, out_channels=output_channel, kernel_size=1, bias=False,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.elu3 = nn.ELU()

        self.stage = nn.Sequential(
            self.conv1, self.bn1, self.elu1,
            self.conv2, self.bn2, self.elu2,
            self.conv3, self.bn3, self.elu3
        )

    def forward(self, x):
        return self.stage(x)


class SimpleNetV2(nn.Module):
    def __init__(self, inpt_size=72, input_channel=3):
        super(SimpleNetV2, self).__init__()
        # (3, 72, 72)
        self.stage_1 = Block(inpt_channel=input_channel, output_channel=128)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (128, 36, 36)
        self.stage_2 = Block(inpt_channel=128, output_channel=256)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (512, 18, 18)
        self.stage_3 = Block(inpt_channel=256, output_channel=512)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.avg_pool3 = nn.AvgPool2d(5)
        # (1024, 9, 9)
        self.stage_4 = Block(inpt_channel=512, output_channel=1024)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.avg_pool4 = nn.AvgPool2d(3)
        # (1024, 4, 4)
        self.stage_5 = Block(inpt_channel=1024, output_channel=2048)
        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (2048, 2, 2)
        self.avg_pool5 = nn.AvgPool2d(2)

        self.elu5 = nn.ELU()

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 17)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # stage 1
        x = self.stage_1(x)
        x = self.maxpool_1(x)

        # stage 2
        x = self.stage_2(x)
        x = self.maxpool_2(x)

        # stage 3
        x = self.stage_3(x)
        x = self.maxpool_3(x)
        residual_3 = self.conv3(x)
        residual_3 = self.avg_pool3(residual_3)

        # stage 4
        x = self.stage_4(x)
        x = self.maxpool_4(x)
        residual_4 = self.conv4(x)
        residual_4 = self.avg_pool4(residual_4)
        # stage 5
        x = self.stage_5(x)
        x = self.maxpool_5(x)

        x = self.avg_pool5(x)

        x = x + residual_3 + residual_4
        x = self.elu5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


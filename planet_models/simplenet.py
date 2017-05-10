from torch.nn import *
from torch.nn import functional as F


class MultiLabelCNN(Module):
    def __init__(self, num_labels, dropout_rate=0.3):
        super(MultiLabelCNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, padding=1, kernel_size=3, bias=False)
        self.bn_conv1 = BatchNorm2d(32)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, padding=1, kernel_size=3, bias=False)
        self.bn_conv2 = BatchNorm2d(64)
        self.conv3 = Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3, bias=False)
        self.bn_conv3 = BatchNorm2d(128)
        self.conv4 = Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3, bias=False)
        self.bn_conv4 = BatchNorm2d(128)
        # Here comes the fully connected layer
        # Because we are using max pooling after each conv operation,
        # the dimension of the last conv layer should be 16*16*128=32768
        self.fc1 = Linear(in_features=32768, out_features=256, bias=False)
        self.fc1_bn = BatchNorm1d(256)
        self.fc1_dropout = Dropout(dropout_rate)
        self.fc2 = Linear(in_features=256, out_features=num_labels)

        for m in self.modules():
            if isinstance(m, Conv2d):
                m.weight.data.normal_(0, 0.02)
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # conv1 computation
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.elu(x)
        x = F.max_pool2d(x, 2, stride=2)


        # conv2 computation
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.elu(x)
        x = F.max_pool2d(x, 2, stride=2)

        # conv3 computation
        x = self.conv3(x)
        x = self.bn_conv3(x)
        x = F.elu(x)
        x = F.max_pool2d(x, 2, stride=2)

        # conv 4 computation
        x = self.conv4(x)
        x = self.bn_conv4(x)
        x = F.elu(x)
        x = F.max_pool2d(x, 2, stride=2)

        # fc1
        x = x.view(-1, 32768)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.elu(x)
        x = self.fc1_dropout(x)

        # output
        x = self.fc2(x)
        return x

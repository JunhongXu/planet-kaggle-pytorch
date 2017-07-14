import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from planet_models.resnet_planet import resnet50_planet, resnet101_planet, resnet152_planet
from planet_models.densenet_planet import densenet201, densenet169, densenet121, densenet161

models_names = [resnet50_planet, resnet101_planet, resnet152_planet, densenet121, densenet161, densenet169, densenet201]


class Blender(nn.Module):
    def __init__(self):
        """
            A blender class blends resnet50, resnet101, resnet152,
            and all densenets output logits to a fully connected layer.
        """
        super(Blender, self).__init__()
        self.models = []
        for m in models_names:
            name = str(m).split()[1]
            model = m().cuda()
            model.eval()

            model = nn.DataParallel(model)
            model.load_state_dict(torch.load('../models/full_data_{}.pth'.format(name)))
            for p in model.parameters():
                p.requires_grad = False
            self.models.append(model)
        self.weighing = nn.Sequential(
            nn.BatchNorm1d(len(models_names)*17),
            nn.Linear(in_features=len(models_names)*17, out_features=17)
        )

    def forward(self, x):
        logits = []
        inference_x = Variable(x.data, volatile=True)
        for m in self.models:
            # make the network in inference mode
            l=m(inference_x)
            # l = F.relu(m(x))       # do we need this?
            logits.append(l)
        logits = Variable(torch.cat(logits, 1).data)
        logits = logits.view(-1, len(models_names) * 17)
        logits = self.weighing(logits)
        return logits


if __name__ == '__main__':
    b = Blender().cuda()
    v = Variable(torch.randn(128, 3, 256, 256)).cuda()
    gt = Variable(torch.randn(128, 17)).cuda()
    pred =  b(v)
    loss = pred - gt
    loss = torch.mean(loss)
    loss.backward()
    for p in b.parameters():
        print(p.grad)
    print(pred)
    # print(b(v))

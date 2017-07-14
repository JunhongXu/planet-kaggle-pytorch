import torch.nn as nn
import torch
from torch.nn import functional as F
from planet_models.resnet_planet import resnet50_planet, resnet101_planet, resnet152_planet
from planet_models.densenet_planet import densenet201, densenet169, densenet121, densenet161

models_names = [resnet152_planet, densenet121, densenet161, densenet169, densenet201]


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
        self.weighing = nn.Linear(in_features=len(models_names)*17, out_features=17)

    def forward(self, x):
        logits = []
        for m in self.models:
            l=m(x)
            # l = F.relu(m(x))       # do we need this?
            logits.append(l)
        logits = torch.cat(logits, 1)
        logits = logits.view(-1, len(models_names) * 17)
        logits = self.weighing(logits)
        return logits


if __name__ == '__main__':
    from torch.autograd import Variable
    b = Blender().cuda()
    for m in b.models:
        for p in m.parameters():
            print(p.requires_grad)
    v = Variable(torch.randn(256, 3, 256, 256), volatile=True).cuda()
    print(b(v))
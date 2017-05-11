from util import optimize_threshold
from planet_models.resnet_planet import resnet34_planet

model = resnet34_planet()
optimize_threshold(model, '../models/resnet-34.pth')

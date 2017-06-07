import pandas as pd
from torch.nn import functional as F
from torchvision.transforms import *
from util import BEST_THRESHOLD
from datasets import test_jpg_loader, mean, std
from labels import *
from planet_models.simplenet_v3 import SimpleNetV3
from planet_models.densenet_planet import densenet121,densenet169
from planet_models.resnet_planet import *
from trainers.train_simplenet import evaluate


def test(models, datasets):
    name = 'ensembles_simple_v3.1_pretrained_densenet121'
    imid_to_label = {}
    for batch_idx, data in enumerate(zip(*datasets)):
        output = 0.0
        for index, (image, im_ids) in enumerate(data):
            output += F.sigmoid(evaluate(models[index], image))

        output = output/len(models)
        output = output.data.cpu().numpy()
        for r, id in zip(output, im_ids):
            label = np.zeros_like(r)
            for i in range(17):
                label[i] = (r[i] > BEST_THRESHOLD[i]).astype(np.int)
            label = np.where(label == 1)[0]
            labels = [idx_to_label[index] for index in label]
            if len(r) == 0:
                print('id', id)
                print('r', r)

            imid_to_label[id] = sorted(labels)
        print('Batch Index {}'.format(batch_idx))
    sample_submission = pd.read_csv('/media/jxu7/BACK-UP/Data/AmazonPlanet/sample_submission.csv')
    for key, value in imid_to_label.items():
        sample_submission.loc[sample_submission['image_name'] == key,'tags'] = ' '.join(str(x) for x in value)
    sample_submission.to_csv('submissions/%s.csv' % name, index=False)


if __name__ == '__main__':
    model1 = nn.DataParallel(densenet169(pretrained=False).cuda())
    model1.load_state_dict(torch.load('models/pretrained_densenet169_wd_1e-4.pth'))
    model1.eval()
    model2 = nn.DataParallel(SimpleNetV3().cuda())
    model2.load_state_dict(torch.load('models/simplenet_v3.1.pth'))
    model2.eval()
    models = [model1, model2]

    datasets = [
        test_jpg_loader(512, transform=Compose([
            Scale(224),
            ToTensor(),
            Normalize(mean, std)
        ])),
        test_jpg_loader(
         512, transform=Compose(
             [
                 Scale(72),
                 ToTensor(),
                 Normalize(mean, std)
             ]))
        ]
    test(models, datasets)

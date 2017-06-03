import pandas as pd
from torch.nn import functional as F
from torchvision.transforms import *
from util import BEST_THRESHOLD
from datasets import test_jpg_loader, mean, std
from labels import *
from planet_models.simplenet_v3 import SimpleNetV3
from trainers.train_densenet import densenet121
from planet_models.resnet_planet import *
from trainers.train_simplenet import evaluate


SIMPLENET = 'models/simplenet_v3.1.pth'
RESNET = 'models/pretrained_densenet121.pth'

def test():
    resnet = nn.DataParallel(densenet121().cuda())
    resnet.load_state_dict(torch.load(RESNET))
    resnet.eval()

    simple_v2 = nn.DataParallel(SimpleNetV3().cuda())
    simple_v2.load_state_dict(torch.load(SIMPLENET))
    simple_v2.eval()

    name = 'ensembles_simple_v3.1_pretrained_densenet121'
    resnet_loader = test_jpg_loader(512, transform=Compose(
        [
            Scale(224),
            ToTensor(),
            Normalize(mean, std)
        ]
    ))

    simple_v2_loader = test_jpg_loader(512, transform=Compose(
        [
            Scale(72),
            ToTensor(),
            Normalize(mean, std)
        ]
    ))



    imid_to_label = {}
    for batch_idx, ((resnet_img, resim_ids), (simplenet_img, testim_ids)) in enumerate(zip(resnet_loader, simple_v2_loader)):
        resnet_result = evaluate(resnet, resnet_img)
        # resnet_result = F.sigmoid(resnet_result)
        # resnet_result = resnet_result.data.cpu().numpy()

        simplenet_result = evaluate(simple_v2, simplenet_img)
        # simplenet_result = F.sigmoid(simplenet_result)

        result = F.sigmoid((simplenet_result + resnet_result) / 2)
        result = result.data.cpu().numpy()
        for r, id in zip(result, testim_ids):
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
    test()

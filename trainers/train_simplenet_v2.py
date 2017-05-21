from torch.nn import *
from util import *
from torch import optim
from planet_models.simplenet_v2 import SimpleNetV2
from datasets import *
import torch


name = 'simplenet_v2'
is_cuda_availible = torch.cuda.is_available()


def train_simplenet_v2_forest(epoch=50):
    criterion = MultiLabelSoftMarginLoss()
    net = SimpleNetV2()
    logger = Logger('../log/', name)
    optimizer = optim.Adam(lr=1e-4, params=net.parameters(), weight_decay=1e-5)
    net.cuda()
    resnet = torch.nn.DataParallel(net, device_ids=[0, 1])
    train_data_set = train_jpg_loader(256, transform=Compose(
        [
            RandomHorizontalFlip(),
            Scale(80),
            RandomCrop(72),
            ToTensor(),
            Normalize(mean, std)
        ]
    ))
    validation_data_set = validation_jpg_loader(64, transform=Compose(
        [
            Scale(72),
            ToTensor(),
            Normalize(mean, std)
         ]
    ))
    best_loss = np.inf
    patience = 0
    for i in range(epoch):
        # training
        for batch_index, (target_x, target_y) in enumerate(train_data_set):
            if is_cuda_availible:
                target_x, target_y = target_x.cuda(), target_y.cuda()
            resnet.train()
            target_x, target_y = Variable(target_x), Variable(target_y)
            optimizer.zero_grad()
            output = resnet(target_x)
            loss = criterion(output, target_y)
            loss.backward()
            optimizer.step()
            if batch_index % 50 == 0:
                print('Training loss is {}'.format(loss.data[0]))
        print('Finished epoch {}'.format(i))

        # evaluating
        val_loss = 0.0
        f2_scores = 0.0
        resnet.eval()
        for batch_index, (val_x, val_y) in enumerate(validation_data_set):
            if is_cuda_availible:
                val_y = val_y.cuda()
            val_y = Variable(val_y, volatile=True)
            val_output = evaluate(resnet, val_x)
            val_loss += criterion(val_output, val_y)
            binary_y = threshold_labels(val_output.data.cpu().numpy())
            f2 = f2_score(val_y.data.cpu().numpy(), binary_y)
            f2_scores += f2
        val_loss = val_loss.data[0]/batch_index
        if best_loss > val_loss:
            print('Saving model...')
            best_loss = val_loss
            torch.save(resnet.state_dict(), '../models/{}.pth'.format(name))
            patience = 0
        else:
            patience += 1
            print('Patience: {}'.format(patience))
            print('Best loss {}, previous loss {}'.format(best_loss, val_loss))

        print('Evaluation loss is {}, Training loss is {}'.format(val_loss, loss.data[0]))
        print('F2 Score is %s' % (f2_scores/batch_index))

        if patience >= 20:
            print('Early stopping!')
            break


        logger.add_record('train_loss', loss.data[0])
        logger.add_record('evaluation_loss', val_loss)
        logger.add_record('f2_score', f2_scores/batch_index)
    logger.save()
    logger.save_plot()


if __name__ == '__main__':
    train_simplenet_v2_forest(epoch=85)


from torch.nn import *
from util import *
from torch import optim
from torchvision.transforms import *
from planet_models.densenet_planet import densenet121, densenet169

NAME = 'pretrained_densenet169_wd_5e-4_SGD'


class RandomVerticalFLip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


def lr_scheduler(epoch, optimizer):
    if epoch <= 10:
        lr = 1e-1
    elif 10 < epoch <= 30:
        lr = 1e-2
    elif 30 < epoch <= 45:
        lr = 5e-3
    else:
        lr = 1e-3

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_optimizer(model, pretrained=True, lr=5e-5, weight_decay=5e-5):
    if pretrained:
        params = [
            {'params': model.features.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr * 10}
        ]
    else:
        params = [
            {'params': model.features.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr}
        ]
    return optim.Adam(params=params, weight_decay=weight_decay)


def train(epoch):
    criterion = MultiLabelSoftMarginLoss()
    net = densenet169(pretrained=False)
    logger = Logger('../log/', NAME)
    optimizer = optim.SGD(lr=1e-1, params=net.parameters(), weight_decay=5e-4, momentum=0.9, nesterov=True)
    # optimizer = get_optimizer(net, False, 1e-4, 1e-4)
    # optimizer = optim.Adam(params=net.parameters(), lr=5e-4, weight_decay=5e-4)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    # resnet.load_state_dict(torch.load('../models/simplenet_v3.pth'))
    train_data_set = train_jpg_loader(72, transform=Compose(
        [

            Scale(256),
            RandomHorizontalFlip(),
            RandomVerticalFLip(),
            RandomCrop(224),
            RandomRotate(),
            ToTensor(),
            Normalize(mean, std)
        ]
    ))
    validation_data_set = validation_jpg_loader(64, transform=Compose(
        [
            Scale(224),
            ToTensor(),
            Normalize(mean, std)
         ]
    ))
    best_loss = np.inf
    patience = 0
    for i in range(epoch):
        # training
        lr_scheduler(epoch, optimizer)
        training_loss = 0.0
        for batch_index, (target_x, target_y) in enumerate(train_data_set):
            if torch.cuda.is_available():
                target_x, target_y = target_x.cuda(), target_y.cuda()
            net.train()
            target_x, target_y = Variable(target_x), Variable(target_y)
            optimizer.zero_grad()
            output = net(target_x)
            loss = criterion(output, target_y)
            training_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            if batch_index % 50 == 0:
                print('Training loss is {}'.format(loss.data[0]))
        print('Finished epoch {}'.format(i))
        training_loss /= batch_index
        # evaluating
        val_loss = 0.0
        net.eval()
        preds = []
        targets = []
        for batch_index, (val_x, val_y) in enumerate(validation_data_set):
            if torch.cuda.is_available():
                val_y = val_y.cuda()
            val_y = Variable(val_y, volatile=True)
            val_output = evaluate(net, val_x)
            val_loss += criterion(val_output, val_y)
            val_output = F.sigmoid(val_output)
            binary_y = val_output.data.cpu().numpy()
            binary_y[binary_y > 0.2] = 1
            binary_y[binary_y <= 0.2] = 0
            preds.append(binary_y)
            targets.append(val_y.data.cpu().numpy())
        targets = np.concatenate(targets)
        preds = np.concatenate(preds)
        f2_scores = f2_score(targets, preds)
        val_loss = val_loss.data[0]/batch_index
        if best_loss > val_loss:
            print('Saving model...')
            best_loss = val_loss
            torch.save(net.state_dict(), '../models/{}.pth'.format(NAME))
            patience = 0
        else:
            patience += 1
            print('Patience: {}'.format(patience))
            print('Best loss {}, previous loss {}'.format(best_loss, val_loss))

        print('Evaluation loss is {}, Training loss is {}'.format(val_loss, training_loss))
        print('F2 Score is %s' % (f2_scores))

        logger.add_record('train_loss', training_loss)
        logger.add_record('evaluation_loss', val_loss)
        logger.add_record('f2_score', f2_scores)
        logger.save()
        logger.save_plot()


if __name__ == '__main__':
    train(200)



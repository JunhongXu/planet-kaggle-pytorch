from torch.nn import *
from util import *
from torch import optim
from torchvision.transforms import *
from planet_models.fpn import FPNet, Bottleneck

NAME = 'fpnet62_wd_1e-4_adam_rotate'


class RandomVerticalFLip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotate(object):
    def __call__(self, img):
        if random.random() < 0.5:
            rotation = np.random.randint(1, 90)
            img = img.rotate(rotation)
        return img


def get_optimizer(model, pretrained=True, lr=5e-5, weight_decay=5e-5):
    if pretrained:
        # no pretrained yet
        pass
    else:
        params = model.parameters()
    return optim.Adam(params=params, lr=lr, weight_decay=weight_decay)


def lr_schedule(epoch, optimizer):
    if epoch < 10:
        lr = 5e-4
    elif 10 <= epoch <= 20:
        lr = 1e-4
    elif 25 < epoch <= 45:
        lr = 5e-5
    else:
        lr = 1e-5

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    criterion = MultiLabelSoftMarginLoss()
    net = FPNet(Bottleneck, [2, 8, 10, 2], dropout_rate=0.4)
    logger = Logger('../log/', NAME)
    # optimizer = get_optimizer(net, False, 1e-4, 5e-4)
    optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    train_data_set = train_jpg_loader(128, transform=Compose(
        [

            Scale(77),
            RandomHorizontalFlip(),
            RandomVerticalFLip(),
            RandomRotate(),
            RandomCrop(72),
            ToTensor(),
            Normalize(mean, std)
        ]
    ))
    validation_data_set = validation_jpg_loader(128, transform=Compose(
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
        training_loss = 0.0
        # adjust learning rate
        lr_schedule(epoch, optimizer)
        for batch_index, (target_x, target_y) in enumerate(train_data_set):
            if torch.cuda.is_available():
                target_x, target_y = target_x.cuda(), target_y.cuda()
            net.train()
            target_x, target_y = Variable(target_x), Variable(target_y)
            optimizer.zero_grad()
            output, prob = net(target_x)
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
            val_output, val_prob = evaluate(net, val_x)
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

        # save for every epoch
        logger.save()
        logger.save_plot()


if __name__ == '__main__':
    train(200)



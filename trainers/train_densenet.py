from torch.nn import *
from util import *
from torch import optim
from torchvision.transforms import *
from planet_models.densenet_planet import densenet121


NAME = 'pretrained_densenet121'


class RandomVerticalFLip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    params = [
        {'params': model.features.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr * 10}
    ]
    return optim.Adam(params=params, weight_decay=weight_decay)


def train(epoch):
    criterion = MultiLabelSoftMarginLoss()
    net = densenet121()
    logger = Logger('../log/', NAME)
    # optimizer = optim.Adam(lr=5e-4, params=net.parameters())
    optimizer = get_optimizer(net)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    # resnet.load_state_dict(torch.load('../models/simplenet_v3.pth'))
    train_data_set = train_jpg_loader(100, transform=Compose(
        [

            Scale(256),
            RandomHorizontalFlip(),
            RandomVerticalFLip(),
            RandomCrop(224),
            ToTensor(),
            Normalize(mean, std)
        ]
    ))
    validation_data_set = validation_jpg_loader(64, transform=Compose(
        [
            Scale(256),
            ToTensor(),
            Normalize(mean, std)
         ]
    ))
    best_loss = np.inf
    patience = 0
    for i in range(epoch):
        # training
        # lr_scheduler(optimizer, epoch)
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

        if patience >= 20:
            print('Early stopping!')
            break

        logger.add_record('train_loss', loss.data[0])
        logger.add_record('evaluation_loss', val_loss)
        logger.add_record('f2_score', f2_scores)
    logger.save()
    logger.save_plot()


if __name__ == '__main__':
    train(200)



from util import *
from planet_models.simplenet import MultiLabelCNN
from torch.nn import *
from torch import optim
from torch.autograd import Variable
from dataset import *
import torch


def train(epoch):
    train_loader = test_jpg_loader(batch_size=64)
    val_loader = test_jpg_loader(batch_size=128)
    logger = Logger('../log/', 'simplenet')
    criterion = MultiLabelSoftMarginLoss()
    model = MultiLabelCNN(17)
    optimizer = optim.Adam(model.parameters(), lr=8e-4, weight_decay=5e-6)
    if torch.cuda.is_available():
        model.cuda()
    for e in range(epoch):
        torch.save(model.state_dict(), 'models/net.pth')
        for batch_idx, (image, target) in enumerate(train_loader):
            model.train()
            if torch.cuda.is_available():
                image, target = image.cuda(), target.cuda()
            image, target = Variable(image), Variable(target)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                model.eval()
                eval_loss = 0
                f2_scores = 0
                for eval_batch_idx, (eval_image, eval_target) in enumerate(val_loader):
                    if torch.cuda.is_available():
                        target = target.cuda()
                    target = Variable(target, volatile=True)
                    output = evaluate(model, eval_image)
                    eval_loss += criterion(output, eval_target)
                    binary_y = threshold_labels(output.data.cpu().numpy())
                    f2 = f2_score(eval_target.data.cpu().numpy(), binary_y)
                    f2_scores += f2
                print('Evaluation loss is {}, training loss is {}'.format(eval_loss.data[0]/eval_batch_idx,
                                                                          loss.data[0]))
                print('F2 score {}'.format(f2_scores/eval_batch_idx))
                logger.add_record('train_loss', loss.data[0])
                logger.add_record('evaluation_loss', eval_loss.data[0]/eval_batch_idx)
                logger.add_record('f2_score', f2_scores/eval_batch_idx)
        print('Finished epoch {}'.format(e))
    logger.save()
    logger.save_plot()



if __name__ == '__main__':
    train(100)

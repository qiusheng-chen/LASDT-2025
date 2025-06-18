import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def test(test_loader, models, len_tar_dataset, device, logger):
    G, F1, F2= models
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct = 0
    correct2 = 0
    size = 0
    # device = G.device
    predict = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)
    pred1_list, pred2_list, label_list, outputdata = [], [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target2 = target
            data1, target1 = Variable(data), Variable(target2)
            output = G(data1)
            outputdata.append(output.cpu().numpy())
            output1 = F1(output)
            output2 = F2(output)
            test_loss += F.nll_loss(output1, target1.long()).item()

            pred1 = output1.data.max(1)[1]  # get the index of the max log-probability
            correct += pred1.eq(target1.data).cpu().sum()
            pred2 = output2.data.max(1)[1]  # get the index of the max log-probability
            correct2 += pred2.eq(target1.data).cpu().sum()
            k = target1.data.size()[0]
            pred1_list.append(pred1.cpu().numpy())
            pred2_list.append(pred2.cpu().numpy())
            predict = np.append(predict, pred1.cpu().numpy())
            labels = np.append(labels, target.cpu().numpy())
            label_list.append(target2.cpu().numpy())
            size += k
            acc1 = 100. * float(correct) / float(size)
            acc2 = 100. * float(correct2) / float(size)

        test_loss = test_loss
        test_loss /= len(test_loader)  # loss function already averages over batch size
        logger.log('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) ({:.2f}%)\n'.format(
            test_loss, correct, len_tar_dataset,
            100. * correct / len_tar_dataset, 100. * correct2 / len_tar_dataset))
        # if 100. * correct / size > 67 or 100. * correct2 / size > 67:
        value = max(100. * correct / len_tar_dataset, 100. * correct2 / len_tar_dataset)

    return value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata, predict, labels

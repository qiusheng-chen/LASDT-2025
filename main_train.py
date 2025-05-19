from __future__ import print_function
import time
import torch.optim as optim
from utils import utils
from utils.utils import *
from models.classifier import *
from evaluation.pre import test
from utils.logger import Logger
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from datetime import datetime, timezone, timedelta
from common.train import *
from training.mcd_train import train
import torch.optim.lr_scheduler as lr_scheduler
from models.backbone import generator
# if __name__ == '__main__':
    # acc_test_list1 = np.zeros([args.num_trials, args.num_trials])
    # acc_test_list2 = np.zeros([args.num_trials, args.num_trials])

if (os.getcwd().split('/')[-1] != 'TAADA'):
    os.chdir(os.path.dirname(os.getcwd()))
''
for iDataSet in range(nDataSet):
    cp_dir = os.path.join(grandparent_dir, 'checkpoints', f'seed{seeds[iDataSet]}')
    heatmap_dir = os.path.join(grandparent_dir, 'heatmap', f'seed{seeds[iDataSet]}')
    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    logger = Logger(cp_dir, local_rank=args.local_rank, log_fn=args.log_fn)
    logger.log(args)
    data_path_s = '/root/data/Projects/DA_240515/data/Houston/Houston13.mat'
    label_path_s = '/root/data/Projects/DA_240515/data/Houston/Houston13_7gt.mat'
    data_path_t = '/root/data/Projects/DA_240515/data/Houston/Houston18.mat'
    label_path_t = '/root/data/Projects/DA_240515/data/Houston/Houston18_7gt.mat'

    data_s, label_s = utils.load_data_houston(data_path_s, label_path_s)
    data_t, label_t = utils.load_data_houston(data_path_t, label_path_t)
    print('#######################idataset######################## ', iDataSet, 'seed', seeds[iDataSet])
    utils.seed_everything(seeds[iDataSet], use_deterministic=True)
    g_train.manual_seed(seeds[iDataSet])
    g_train_tar.manual_seed(seeds[iDataSet])
    g_test.manual_seed(seeds[iDataSet])
    g_source_test.manual_seed(seeds[iDataSet])

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, Gr, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)
    testID_s, testX_s, testY_s, Gr_s, RandPerm_s, Row_s, Column_s = utils.get_all_data(data_s, label_s, HalfWidth)

    test_source_dataset = TensorDataset(torch.tensor(testX_s), torch.tensor(testY_s))
    train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,worker_init_fn=seed_worker,
    generator=g_train,)
    train_tar_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, worker_init_fn=seed_worker,
    generator=g_train_tar,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, worker_init_fn=seed_worker,
    generator=g_test,)
    source_test_loader = DataLoader(test_source_dataset, batch_size=batch_size, shuffle=False, drop_last=False, worker_init_fn=seed_worker,
    generator=g_source_test,)
    # train_loader = source_test_loader

    len_src_loader = len(train_loader)
    len_tar_train_loader = len(train_tar_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_train_dataset = len(train_tar_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)
    ##########################################################################################################
    F1 = ResClassifier(out_dim=num_classes, in_dim= args.spe_dim + args.spa_dim, hidden_dim=args.f_hidden_dim)
    F2 = ResClassifier(out_dim=num_classes, in_dim= args.spe_dim + args.spa_dim, hidden_dim=args.f_hidden_dim)

    G = generator(input_channels=N_BANDS, patch_size=patch_size, spe_dim=args.spe_dim, spe_patch_size=args.spe_patch_size,
                  spa_dim = args.spa_dim,
                  depth=args.tf_depth, num_classes=num_classes, stride=args.stride,
                  heads=args.heads, dim_head=args.dim_head, dropout=args.tf_dropout,
                  emb_dropout=args.emb_dropout)
    F1.apply(weights_init)
    F2.apply(weights_init)

    models = (G, F1, F2)
    # add a scheduler
    # lr = args.lr
    if args.cuda:
        G.to(DEV)
        F1.to(DEV)
        F2.to(DEV)

    if args.optimizer == 'momentum':
        optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr ,
                                weight_decay=0.0005)

    elif args.optimizer == 'adam':
        optimizer_g = optim.Adam(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
        lr_decay_gamma = 0.3

    else:
        optimizer_g = optim.Adadelta(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adadelta(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
        lr_decay_gamma = 0.3
    if args.lr_scheduler == 'cosine':
        scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer_g, args.num_epoch)
        scheduler_f = lr_scheduler.CosineAnnealingLR(optimizer_f, args.num_epoch)

    elif args.lr_scheduler == 'step_decay':
        milestones = [int(0.5 * args.num_epoch), int(0.75 * args.num_epoch)]
        scheduler_g = lr_scheduler.MultiStepLR(optimizer_g, gamma=lr_decay_gamma, milestones=milestones)
        scheduler_f = lr_scheduler.MultiStepLR(optimizer_f, gamma=lr_decay_gamma, milestones=milestones)
    else:
        raise NotImplementedError()
    from training.scheduler import GradualWarmupScheduler

    scheduler_warmup_g = GradualWarmupScheduler(optimizer_g, multiplier=args.multiplier, total_epoch=args.warmup,
                                              after_scheduler=scheduler_g)
    scheduler_warmup_f = GradualWarmupScheduler(optimizer_f, multiplier=args.multiplier, total_epoch=args.warmup,
                                                after_scheduler=scheduler_f)

    optimizers = (optimizer_g, optimizer_f)
    schedulers = (scheduler_warmup_g, scheduler_warmup_f)

    train_start = time.time()
    for ep in range(1, num_epoch + 1):
        train(models, optimizers, schedulers, ep, train_loader, train_tar_loader, len_src_loader,
              args, seeds[iDataSet], num_k, logger, DEV)
        # correct = val(val_loader)
        # 5 epoch test
        if ep % args.num_epoch == 0 or ep % args.test_step == 0:
            test_start = time.time()
            value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata_target, predict, labels \
                = (test(test_loader, models, len_tar_dataset, DEV, logger))
            test_end = time.time()
            total_acc[iDataSet] = acc1
            C = metrics.confusion_matrix(labels, predict)
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)
            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
            if ep%args.num_epoch ==0:
                torch.save(F1.state_dict(), os.path.join(cp_dir, f'F1_ep{ep}_seed{seeds[iDataSet]}.pt'))
                torch.save(F2.state_dict(), os.path.join(cp_dir, f'F2_ep{ep}_seed{seeds[iDataSet]}.pt'))
                torch.save(G.state_dict(), os.path.join(cp_dir, f'G_ep{ep}_seed{seeds[iDataSet]}.pt'))
            if acc1 >= best_test_acc:        
                best_test_acc = acc1
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = Gr, RandPerm, Row, Column

    logger.log(f"ACC:{acc1}")
    train_end = time.time()

AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(total_acc)
OAStd = np.std(total_acc)
kMean = np.mean(k)
kStd = np.std(k)
logger.log("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
logger.log("test time per DataSet(s): " + "{:.5f}".format(test_end - test_start))
logger.log("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
logger.log("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
logger.log("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
logger.log("accuracy for each class: ")
for i in range(num_classes):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))
beijing = timezone(timedelta(hours=8))
# cn = timezone(beijing)
now_time = datetime.now(beijing)
# temp = np.random.randint(1, 100, size=(3, 1))
index = np.argsort(total_acc, axis=0).reshape(-1)
total_acc = total_acc[index]
seeds = np.array(seeds)
# seeds = torch.tensor(seeds)
seeds = seeds[index]
# A = A[index]
with open(os.path.join(results_dir, 'results.txt'), "a") as f:
    f.write(f'in {now_time},\tconfig:\t{args.__str__()}\n')
    f.write("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start) + "\n")
    f.write("test time per DataSet(s): " + "{:.5f}".format(test_end - test_start) + "\n")

    f.write("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd) + "\n")
    f.write("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd) + "\n")
    f.write("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd) + "\n")
    f.write("accuracy for each class: " + "\n")
    for i in range(num_classes):
        f.write(
            "Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]) + "\n")
    for i in range(len(seeds)):
        f.write(f'seed {seeds[i]} acc is {total_acc[i]}\n')

print('classification map!!!!!')
for i in range(len(best_predict_all)):
    best_G[best_Row[best_RandPerm[i]]][best_Column[best_RandPerm[ i]]] = best_predict_all[i] + 1

###################################################
hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
predict_hmp_dir = os.path.join(heatmap_dir, 'predict_map')
if not os.path.exists(predict_hmp_dir):
    os.makedirs(predict_hmp_dir)
classification_map(hsi_pic[2:-2, 2:-2, :], best_G[2:-2, 2:-2], 24, os.path.join(predict_hmp_dir, 'houston.png'))
#





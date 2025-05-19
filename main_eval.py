from __future__ import print_function
import time
from utils import utils
from models.classifier import *
from evaluation.pre import test
from torch.utils.data import TensorDataset, DataLoader
from common.eval import *
from utils.logger import Logger
from models.backbone import generator

if (os.getcwd().split('/')[-1] != 'TAADA'):
    os.chdir(os.path.dirname(os.getcwd()))

cp_dir = os.path.join(grandparent_dir, 'checkpoints', f'seed{seed}')

heatmap_dir = os.path.join(grandparent_dir, 'heatmap', args.dataset_name, f'seed{seed}')
if not os.path.exists(cp_dir):
    os.makedirs(cp_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(heatmap_dir):
    os.makedirs(heatmap_dir)
logger = Logger(cp_dir, local_rank=args.local_rank)
logger.log(args)
data_path_s = '/root/data/Projects/DA_240515/data/Houston/Houston13.mat'
label_path_s = '/root/data/Projects/DA_240515/data/Houston/Houston13_7gt.mat'
data_path_t = '/root/data/Projects/DA_240515/data/Houston/Houston18.mat'
label_path_t = '/root/data/Projects/DA_240515/data/Houston/Houston18_7gt.mat'

data_s, label_s = utils.load_data_houston(data_path_s, label_path_s)
data_t, label_t = utils.load_data_houston(data_path_t, label_path_t)

trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
testID, testX, testY, Gr, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)
testID_s, testX_s, testY_s, Gr_s, RandPerm_s, Row_s, Column_s = utils.get_all_data(data_s, label_s, HalfWidth)

test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


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
G.load_state_dict(torch.load(os.path.join(cp_dir, f'G_ep{num_epoch}_seed{seed}.pt')))
F1.load_state_dict(torch.load(os.path.join(cp_dir, f'F1_ep{num_epoch}_seed{seed}.pt')))
F2.load_state_dict(torch.load(os.path.join(cp_dir, f'F2_ep{num_epoch}_seed{seed}.pt')))


models = (G, F1, F2)
# add a scheduler
# lr = args.lr
if args.cuda:
    G.cuda()
    F1.cuda()
    F2.cuda()

test_start = time.time()
value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata_target, predict, labels \
    = (test(test_loader, models, len_tar_dataset, DEV,logger))





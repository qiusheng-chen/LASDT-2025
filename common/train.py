import warnings
warnings.filterwarnings('ignore')
from common.common import args
import torch
import numpy as np
import os
grandparent_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

args.cuda = not args.no_cuda and torch.cuda.is_available()
DEV = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
num_epoch = args.num_epoch
batch_size = args.batch_size
num_k = args.num_k
if args.dataset_name == 'houston':
    num_classes = 7
    N_BANDS = 48
    seeds = [1700, 1764, 2883, 1872, 2153, 2668, 2625, 2738, 2775, 2888]
elif args.dataset_name == 'hyrank':
    num_classes = 12
    N_BANDS = 176
    seeds = [1520, 3046, 4069, 4111, 4315, 4261, 4396, 4406, 4437, 4567]
elif args.dataset_name == 'pavia':
    num_classes = 7
    N_BANDS = 102
    seeds = [1538, 1518, 2046, 2154, 2249, 2351, 2296, 2269, 2268, 2313]


HalfWidth = args.halfwidth

patch_size = 2 * HalfWidth + 1
nDataSet = len(seeds)
total_acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, num_classes])
k = np.zeros([nDataSet, 1])
best_acc_all = 0.0
best_predict_all = 0
best_test_acc = 0
best_source_feature = []
best_source_label = []
best_target_feature = []
best_target_label = []
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

fn = f'lr_{args.lr}'

results_dir = os.path.join(grandparent_dir, 'results')

g_train = torch.Generator()
g_train_tar = torch.Generator()
g_test = torch.Generator()
g_source_test = torch.Generator()


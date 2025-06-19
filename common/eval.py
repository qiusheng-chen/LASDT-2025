from common.common import args
import torch
import os
grandparent_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

seed = 1700
args.cuda = not args.no_cuda and torch.cuda.is_available()
DEV = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
num_epoch = args.num_epoch
# num_epoch = 50
batch_size = args.batch_size
num_k = args.num_k
<<<<<<< HEAD
if args.dataset_name == 'houston':
    num_classes = 7
    N_BANDS = 48
    seeds = [1700, 1764, 2883, 1872, 2153, 2668, 2625, 2738, 2775, 2888]
    HalfWidth = 3
elif args.dataset_name == 'hyrank':
    num_classes = 12
    N_BANDS = 176
    seeds = [1520, 3046, 4069, 4111, 4315, 4261, 4396, 4406, 4437, 4567]
    HalfWidth = 1
elif args.dataset_name == 'pavia':
    num_classes = 7
    N_BANDS = 102
    seeds = [1538, 1518, 2046, 2154, 2249, 2351, 2296, 2269, 2268, 2313]
    HalfWidth = 4


=======
num_classes = 7
N_BANDS = 48
HalfWidth = args.halfwidth
>>>>>>> 0c47779a53348f10cc223db0d97e06fe99582c50
BATCH_SIZE = 32
patch_size = 2 * HalfWidth + 1
results_dir = os.path.join(grandparent_dir, 'results')




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
num_classes = 7
N_BANDS = 48
HalfWidth = args.halfwidth
BATCH_SIZE = 32
patch_size = 2 * HalfWidth + 1
results_dir = os.path.join(grandparent_dir, 'results')




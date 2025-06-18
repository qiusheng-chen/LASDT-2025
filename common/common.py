import argparse

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')

parser.add_argument('--save_step', type=int, default=200, metavar='N',
                    help='number of epochs to save checkpoints (default: 20)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
# parser.add_argument('--seeds', type=int, default=[1700, 1764, 2883, 1872, 2153, 2668, 2625, 2738, 2775, 2888], metavar='K',
#                     help='seeds for training [1700, 1764, 2883, 1872, 2153, 2668, 2625, 2738, 2775, 2888]')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dataset_name', default='houston', type=str)
# distributed training
parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')

#import parameters####################################################################################################
parser.add_argument('--test_step', type=int, default=50, metavar='N',
                    help='number of epochs to test results (default: 20)')
parser.add_argument('--log_fn', default='log.txt', type=str)
parser.add_argument('--halfwidth', default=3, type=int)

parser.add_argument('--spe_patch_size', default=6, type=int, help='patch size of spectral transformer')

parser.add_argument('--spe_dim', default=196, type=int, help='dim for spectral transformer')
parser.add_argument('--spa_dim', default=196, type=int, help='dim for spatial transformer')

parser.add_argument('--f_hidden_dim', default=76, type=int, help='hidden dim of classifier')

parser.add_argument('--tf_depth', default=6, type=int, help='the depth of generator')

parser.add_argument('--tf_dropout', default=0, type=float, help='dropout for generator attention')

parser.add_argument('--emb_dropout', default=0, type=float, help='dropout for generator embeddings')

parser.add_argument('--decoder_dropout', default=0, type=float, help='dropout for decoder attention')

parser.add_argument('--stride', default=8, type=int, help='stride for generate patches')
parser.add_argument('--heads', default=7, type=int, help='heads for transformer')

parser.add_argument('--dim_head', default=128, type=int, help='dim for every head')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--num_epoch', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.025, metavar='LR',
                    help='learning rate (default: 0.02)')

parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
parser.add_argument('--warmup', help='Warm-up epochs 10',
                        default=6 , type=int)

parser.add_argument('--multiplier', help='Multiplier for lr scheduler 1+1e-8',
                        default=1+1e-8, type=int)
args = parser.parse_args()

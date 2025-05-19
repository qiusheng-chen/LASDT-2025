import os
import pickle
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import time

class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, logdir,  local_rank=0, log_fn='log.txt'):
        self.local_rank = local_rank
        self.log_fn = log_fn
        if self.local_rank == 0:
            # if not os.path.exists("./logs/"):
            #     os.mkdir("./logs/")

            # logdir = self._make_dir(fn)
            if not os.path.exists(logdir):
                os.mkdir(logdir)
            # if len(os.listdir(logdir)) != 0:
            #     shutil.rmtree(logdir)

            # if len(os.listdir(logdir)) != 0 and ask:
            #     ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
            #                 "Will you proceed [y/N]? ")
            #     if ans in ['y', 'Y']:
            #         shutil.rmtree(logdir)
            #     else:
            #         pass
                    # exit(1)

            self.set_dir(logdir)

    def _make_dir(self, fn):
        today = datetime.today().strftime("%y%m%d")
        logdir = 'logs/' + fn
        return logdir

    def set_dir(self, logdir):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        file_path = os.path.join(logdir, self.log_fn)
        if os.path.isfile(file_path):
            # 如果文件存在，使用os.remove()方法删除文件
            os.remove(file_path)
        self.log_file = open(file_path, 'a')

    def log(self, string):
        if self.local_rank == 0:
            self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
            self.log_file.flush()

            print('[%s] %s' % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.local_rank == 0:
            self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
            self.log_file.flush()

            print('%s (%s)' % (string, self.logdir))
            sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.local_rank == 0:
            self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        if self.local_rank == 0:
            self.writer.add_image(tag, images, step)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        if self.local_rank == 0:
            self.writer.add_histogram(tag, values, step, bins='auto')


from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
import time
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(models, optimizers, schedulers, ep, train_source_loader, train_tar_loader, len_src_loader,
          args, seed, num_k, logger, device):
    iter_source, iter_target = iter(train_source_loader), iter(train_tar_loader)
    criterion = nn.CrossEntropyLoss().to(device)
    G, F1, F2 = models
    G.train()
    F1.train()
    F2.train()
    optimizer_g, optimizer_f = optimizers
    scheduler_g, scheduler_f= schedulers
    num_iter = len_src_loader
    loss_sf1_list = []
    loss_sf2_list = []
    loss_dis_list = []

    start_time = time.time()

    for batch_idx in range(1, num_iter):
        if batch_idx % len(train_tar_loader) == 0:
            iter_target = iter(train_tar_loader)
        data_source, label_source = next(iter_source)
        data_target, _ = next(iter_target)
        batch_size = data_target.size(0)
        if args.cuda:
            data_s, label_s = data_source.to(device), label_source.to(device)
            data_t = data_target.to(device)
        # when pretraining network source only
        eta = 1.0
        label_s = Variable(label_s)
        data = torch.cat([data_s, data_t], dim=0)
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        output_g = G(data)  #特征映射网络
        output_f1 = F1(output_g)  # 两层全连接网络，第一层relu,第二层没有激活层，输出为类别数
        output_f2 = F2(output_g)
        output_s_f1 = output_f1[:batch_size, :]
        output_s_f2 = output_f2[:batch_size, :]
        output_t_f1 = output_f1[batch_size:, :]
        output_t_f2 = output_f2[batch_size:, :]

        loss_s_f1 = criterion(output_s_f1, label_s.long())
        loss_s_f2 = criterion(output_s_f2, label_s.long())
        all_loss = (loss_s_f1 + loss_s_f2 )
        all_loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        # Step B train classifier to maximize discrepancy
        optimizer_f.zero_grad()
        output_g = G(data)  # 特征映射网络
        # output_g = ST(output_g)
        output_f1 = F1(output_g)  # 两层全连接网络，第一层relu,第二层没有激活层，输出为类别数
        output_f2 = F2(output_g)
        output_s_f1 = output_f1[:batch_size, :]
        output_s_f2 = output_f2[:batch_size, :]
        output_t_f1 = output_f1[batch_size:, :]
        output_t_f2 = output_f2[batch_size:, :]
        output_t_f1 = F.softmax(output_t_f1, dim=1)
        output_t_f2 = F.softmax(output_t_f2, dim=1)
        loss_s_f1 = criterion(output_s_f1, label_s.long())
        loss_s_f2 = criterion(output_s_f2, label_s.long())
        loss_dis = torch.mean(torch.abs(output_t_f1 - output_t_f2))

        F_loss = (loss_s_f1 + loss_s_f2 - eta * loss_dis
                  )
        F_loss.backward()
        optimizer_f.step()

        # Step C train genrator to minimize discrepancy
        for i in range(num_k):
            optimizer_g.zero_grad()
            output_g = G(data)  # 特征映射网络
            # output_g = ST(output_g)
            output_f1 = F1(output_g)  # 两层全连接网络，第一层relu,第二层没有激活层，输出为类别数
            output_f2 = F2(output_g)
            output_s_f1 = output_f1[:batch_size, :]
            output_s_f2 = output_f2[:batch_size, :]
            output_t_f1 = output_f1[batch_size:, :]
            output_t_f2 = output_f2[batch_size:, :]

            output_t_f1 = F.softmax(output_t_f1, dim=1)
            output_t_f2 = F.softmax(output_t_f2, dim=1)
            loss_dis = torch.mean(torch.abs(output_t_f1 - output_t_f2))
            loss_s_f1 = criterion(output_s_f1, label_s.long())
            loss_s_f2 = criterion(output_s_f2, label_s.long())
            loss = loss_dis
            loss.backward()
            optimizer_g.step()
        scheduler_g.step(ep - 1 + (batch_idx - 1) / (len_src_loader))
        scheduler_f.step(ep - 1 + (batch_idx - 1) / (len_src_loader))
        lr = optimizer_g.param_groups[0]['lr']

        if batch_idx % args.log_interval == 0:
            loss_sf1_list.append(loss_s_f1.item())
            loss_sf2_list.append(loss_s_f2.item())
            loss_dis_list.append(loss_dis.item())


        if batch_idx == 1 and ep > 1:
            G.train()
            F1.train()
            F2.train()
    run_time = time.time() - start_time
    logger.log('Training Ep: {} LossSF1: {:.4f} LossSF2: {:.4f} Dis: {:.4f} LR: {:.6f} Time: {:.2f}s'.format(
            ep, sum(loss_sf1_list) / len(loss_sf1_list),
                sum(loss_sf2_list) / len(loss_sf2_list),
                sum(loss_dis_list) / len(loss_dis_list),

                lr,
                run_time))






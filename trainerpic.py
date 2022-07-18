import os
import tqdm
import time
# import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from common.optimizer import ScheduledOptim
from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run
from modelso.dataloader.samplers import CategoriesSampler
from modelso.dataloader.data_utils import dataset_builder
from modelso.mfndc import MFNDC
from test import test_main, evaluate


def train(epoch, model, loader, optimizer, args=None):
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...

    loss_meter = Meter()
    acc_meter = Meter()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)
    #lr调节
    lr_mult = (1 / 1e-5) ** (1 / 100)
    lr = []
    losses = []
    #lr调节
    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):

        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()
        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        data = model(data)
        data_aux = model(data_aux)  # I prefer to separate feed-forwarding data and data_aux due to BN

        # loss for batch
        model.module.mode = 'cca'
        data_shot, data_query = data[:k], data[k:]
        logits, absolute_logits= model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
        epi_loss = F.cross_entropy(logits, label)
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)
        loss_aux = loss_aux + absolute_loss
        loss = args.lamb * epi_loss + loss_aux
        acc = compute_accuracy(logits, label)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description('[train] epo:{:>3} | avg.loss:{:.4f} | avg.acc:{:.3f} (curr:{:.3f})'.format(epoch, loss_meter.avg(),acc_meter.avg(),acc))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        #detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()
        #lr调节
        lr.append(optimizer.learning_rate)
        losses.append(loss.item())
        optimizer.set_learning_rate(optimizer.learning_rate * lr_mult)
        if i==90:
            plt.figure()
            plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
            plt.xlabel('learning rate')
            plt.ylabel('loss')
            plt.plot(np.log(lr), losses)
            time_str = time.strftime("%Y%m%d") + time.strftime("_%H%M%S")
            plt.savefig("picture/" + time_str + "lr_loss" +".png")
            plt.figure()
            plt.xlabel('num iterations')
            plt.ylabel('learning rate')
            plt.plot(lr)
            plt.savefig("picture/" + time_str + "lr_iter" +".png")
    #lr调节


    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def train_main(args):
    Dataset = dataset_builder(args)

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]

    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    set_seed(args.seed)
    model = MFNDC(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    if not args.no_wandb:
        writer = SummaryWriter('./keshilog')
    #     imag=torch.randn(1,3,84,84)
    #     writer.add_graph(model,input_to_model=imag)
        # wandb.watch(model)
    #print(1234,model)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    #lr调节 改210 211 263
    basic_optim = torch.optim.SGD(model.parameters(), lr=1e-5)
    optimizer = ScheduledOptim(basic_optim)
    #lr调节

    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)

    # 构建 SummaryWriter
    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()
        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args)
        val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')
        if epoch==1 or epoch==2 or epoch%10==0:
            test_loss, test_acc, _ = evaluate(epoch, model, test_loader, args, set='test')
        if not args.no_wandb:

            print('Epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'
                  .format(epoch, train_loss, train_acc))

            print('Epoch: {}, Val Loss: {:.6f}, Val Acc: {:.6f}'
                  .format(epoch, val_loss, val_acc))
            #########记录数据，保存于event file，这里记录了每一个epoch的损失和准确度########
            writer.add_scalars("Loss", {"Train": train_loss}, epoch)
            writer.add_scalars("Accuracy", {"Train": train_acc}, epoch)
            writer.add_scalars("Loss", {"Val": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"Val": val_acc}, epoch)
            ############## 每个epoch，记录梯度，权值#######################################
            for name, param in model.named_parameters():      #返回模型的参数
                writer.add_histogram(name + '_grad', param.grad, epoch)   #参数的梯度
                writer.add_histogram(name + '_data', param, epoch)        #参数的权值


            # wandb.log({'train/loss': train_loss, 'train/acc': train_acc, 'val/loss': val_loss, 'val/acc': val_acc}, step=epoch)

        if val_acc > max_acc:
            print('[ log ] *********A better model is found %.3f *********'% val_acc)
            max_acc, max_epoch = val_acc, epoch
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))

        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'epoch_{}.pth'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_epoch_{}.pth'.format(epoch)))

        epoch_time = time.time() - start_time
        print('[ log ] saving @ {}'.format(args.save_path))
        print('[ log ] roughly {:.2f} h left\n'.format((args.max_epoch - epoch) / 3600. * epoch_time))

        lr_scheduler.step()

    return model


if __name__ == '__main__':
    args = setup_run(arg_mode='train')

    model = train_main(args)
    test_acc, test_ci = test_main(model, args)

    if not args.no_wandb:
        print(' test/acc:{}  test/confidence_interval: {}'.format(test_acc,test_ci))

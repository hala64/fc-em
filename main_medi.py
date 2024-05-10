import os
import argparse
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
parser = argparse.ArgumentParser()
parser.set_defaults(entry=lambda cmd_args: parser.print_help())
parser.add_argument('--entry', type=str, default="train")
parser.add_argument('--exp-config', type=str, default="")
parser.add_argument('--model-path', type=str, default="")
parser.add_argument('--resume', action="store_true", default=False)
parser.add_argument('--gpu-id', type=str, default='1',
                    help="Which gpu to use")
parser.add_argument('--output', type=str, default='./test.txt',
                    help="path to output file")
parser.add_argument('--severity', type=int, default=1,
                    help="Which severity to use")
parser.add_argument('--pgd_step', type=int, default=20,
                    help="Which severity to use")
parser.add_argument('--eps', type=float, default=1.0,
                    help="Which severity to use")
parser.add_argument('--confusion', action="store_true", default=False,
                    help="whether to output confusion matrix data")
parser.add_argument('--fixed-beta', action="store_true", default=False,
                    help="whether using fixed beta for a part of dataset")
parser.add_argument('--fixed-beta-ratio', type=float, default=0.0,
                    help="ratio of using fixed beta")
parser.add_argument('--temperature', type=float, default=0.1,
                    help="temperature of similarity loss")
parser.add_argument('--chamf_coeff', type=float, default=3.0,
                    help="chamfer distance coefficient")
parser.add_argument('--zeta', type=float, default=1.0,
                    help="strength of ce loss when conducting fd-ap(-t) attack")
parser.add_argument('--threshold', type=float, default=-1.0,
                    help="threshold of adv loss when conducting fc-em-thre")
parser.add_argument('--expand-coeff', type=float, default=10.0,
                    help="expand coefficient when conducting fc-em-thre")
parser.add_argument('--poison-ratio', type=float, default=1.0,
                    help="ratio of the poisoned dataset")
parser.add_argument('--step-size', type=float, default=0.015,
                    help="step size when generating poisons")
parser.add_argument('--steps', type=int, default=10,
                    help="steps of generating poisons")
parser.add_argument('--change-point', action="store_true", default=False,
                    help="change the number of point for each point cloud")
parser.add_argument('--warmup', type=int, default=0,
                    help="warmup epochs when conducting adaptive beta")

cmd_args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = cmd_args.gpu_id
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random
from dataloader import create_dataloader
from time import time
from datetime import datetime
from progressbar import ProgressBar
import models
from torch.autograd import Variable
from collections import defaultdict
import numpy as np
from all_utils import (
    TensorboardManager, PerfTrackTrain,
    PerfTrackVal, TrackTrain, smooth_loss, DATASET_NUM_CLASS)
from configs import get_cfg_defaults
import pprint
from pointnet_pyt.pointnet.model import feature_transform_regularizer
import aug_utils


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    print('WARNING: Using CPU')

os.environ['CUDA_VISIBLE_DEVICES'] = cmd_args.gpu_id


def check_inp_fmt(task, data_batch, dataset_name):
    if task in ['cls', 'cls_trans']:
        pc, label = data_batch['pc'], data_batch['label']
        if dataset_name == 'modelnet40_dgcnn':
            assert isinstance(pc, torch.FloatTensor) or isinstance(
                pc, torch.DoubleTensor)
        else:
            if not isinstance(pc, torch.FloatTensor):
                pc = pc.float()
            assert isinstance(pc, torch.FloatTensor)
        assert isinstance(label, torch.LongTensor)
        assert len(pc.shape) == 3
        assert len(label.shape) == 1
        b1, _, y = pc.shape[0], pc.shape[1], pc.shape[2]
        b2 = label.shape[0]
        assert b1 == b2
        assert label.max().item() < DATASET_NUM_CLASS[dataset_name]
        assert label.min().item() >= 0
    else:
        assert NotImplemented


def check_out_fmt(task, out, dataset_name):
    if task == 'cls':
        logit = out['logit']
        assert isinstance(logit, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert len(logit.shape) == 2
        assert DATASET_NUM_CLASS[dataset_name] == logit.shape[1]
    elif task == 'cls_trans':
        logit = out['logit']
        trans_feat = out['trans_feat']
        assert isinstance(logit, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert isinstance(trans_feat, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert len(logit.shape) == 2
        assert len(trans_feat.shape) == 3
        assert trans_feat.shape[0] == logit.shape[0]
        assert (trans_feat.shape[1] == trans_feat.shape[2]) and (trans_feat.shape[1] == 64)
        assert DATASET_NUM_CLASS[dataset_name] == logit.shape[1]
    else:
        assert NotImplemented


def get_inp(cfg, task, model, data_batch, batch_proc, dataset_name, poison=None):
    check_inp_fmt(task, data_batch, dataset_name)
    if not batch_proc is None:
        data_batch = batch_proc(data_batch, DEVICE)
        check_inp_fmt(task, data_batch, dataset_name)

    if task in ['cls', 'cls_trans']:
        pc = data_batch['pc']
        inp = {'pc': pc}
    else:
        assert False

    return inp


def get_loss(cfg, task, loss_name, data_batch, out, dataset_name, label_bias=False, poison=None):
    check_out_fmt(task, out, dataset_name)
    if task == 'cls':
        if label_bias:
            label = ((data_batch['label'] + 3) % 40).to(out['logit'].device)

        from all_utils import chamfer_loss_standard
        adv_loss = F.cross_entropy(out['logit'], label)
        dist_loss = chamfer_loss_standard(data_batch, poison)
        loss = adv_loss + cmd_args.chamf_coeff * dist_loss

    elif task == 'cls_trans':
        if label_bias:
            label = ((data_batch['label'] + 3) % 40).to(out['logit'].device)
        else:
            label = data_batch['label'].to(out['logit'].device)
        trans_feat = out['trans_feat']
        if loss_name == 'cross_entropy':
            loss = F.cross_entropy(out['logit'], label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        elif loss_name == 'smooth':
            loss = smooth_loss(out['logit'], label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        from all_utils import chamfer_loss_standard
        adv_loss = F.cross_entropy(out['logit'], label)
        dist_loss = chamfer_loss_standard(data_batch, poison)
        loss = adv_loss + cmd_args.chamf_coeff * dist_loss
    else:
        assert False

    return loss

def validate(task, loader, model, dataset_name, confusion = False):
    model.eval()

    def get_extra_param():
        return None

    perf = PerfTrackVal(task, extra_param=get_extra_param())
    time_dl = 0
    time_gi = 0
    time_model = 0
    time_upd = 0

    with torch.no_grad():
        time5  = time()
        if confusion:
            pred = []
            ground = []
        for i, data_batch_ori in enumerate(loader):
            data_batch = {
                'pc': data_batch_ori[0][:, :, :3],
                'label': data_batch_ori[1].squeeze(),
                'index': data_batch_ori[2]
            }
            time1 = time()
            loader_dataset_batch_proc = None
            inp = get_inp(cfg, task, model, data_batch, loader_dataset_batch_proc, dataset_name)
            time2 = time()

            out = model(**inp)

            if confusion:
                pred.append(out['logit'].squeeze().cpu())
                ground.append(data_batch['label'].squeeze().cpu())

            time3 = time()
            perf.update(data_batch=data_batch, out=out)
            time4 = time()

            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()

    print(f"Time DL: {time_dl}, Time Get Inp: {time_gi}, Time Model: {time_model}, Time Update: {time_upd}")
    if not confusion:
        return perf.agg()
    else:
        pred = np.argmax(torch.cat(pred).numpy(), axis=1)
        ground = torch.cat(ground).numpy()
        return perf.agg(), pred, ground

def train(task, loader, model, optimizer, loss_name, dataset_name, cfg, poison, loss_stat=None):
    model.train()

    def get_extra_param():
       return None

    perf = PerfTrackTrain(task, extra_param=get_extra_param())
    time_forward = 0
    time_backward = 0
    time_data_loading = 0

    time3  = time()

    label_bias = False

    for i, data_batch_ori in enumerate(loader):
        if cfg.EXP.DATASET == 'medpt':
            real_batch_size = 25
            choice = np.random.choice(len(data_batch_ori[0]), len(data_batch_ori[0]), replace=False)
            index_full = choice.reshape(-1, real_batch_size)
            for j in range(len(index_full)):
                index = index_full[j]
                data_batch = {
                    'pc': data_batch_ori[0][index][:, :, :3],
                    'label': data_batch_ori[1][index].squeeze(),
                    'index': index
                }
                time1 = time()
                x_natural = data_batch.copy()

                if cfg.AUG.NAME == 'cutmix_r':
                    data_batch = aug_utils.cutmix_r(data_batch, cfg)
                elif cfg.AUG.NAME == 'cutmix_k':
                    data_batch = aug_utils.cutmix_k(data_batch, cfg)
                elif cfg.AUG.NAME == 'mixup':
                    data_batch = aug_utils.mixup(data_batch, cfg)
                elif cfg.AUG.NAME == 'rsmix':
                    data_batch = aug_utils.rsmix(data_batch, cfg)
                elif cfg.AUG.NAME == 'pgd':
                    clip_func = None
                    data_batch = aug_utils.pgd(cfg, clip_func, data_batch, model, task, loss_name, dataset_name)
                    model.train()
                index = data_batch['index']
                loader_dataset_batch_proc = None
                inp = get_inp(cfg, task, model, data_batch, loader_dataset_batch_proc, dataset_name)

                out = model(inp['pc'].cuda() + poison[index])

                loss = get_loss(cfg, task, loss_name, data_batch, out, dataset_name, label_bias=label_bias,
                                poison=poison)

                perf.update_all(data_batch=data_batch, out=out, loss=loss)
                time2 = time()

                ## robust loss
                if cfg.ADVT.NAME == 'trades':
                    criterion_kl = nn.KLDivLoss(size_average=False)
                    batch_size = len(data_batch['pc'])
                    data_batch['pc'] = Variable(data_batch['pc'], requires_grad=False)
                    loader_dataset_batch_proc = None
                    natural_inp = get_inp(cfg, task, model, x_natural, loader_dataset_batch_proc, dataset_name)
                    adv_inp = get_inp(cfg, task, model, data_batch, loader_dataset_batch_proc, dataset_name)
                    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(**adv_inp)['logit'], dim=1),
                                                                    F.softmax(model(**natural_inp)['logit'], dim=1))
                    loss = loss + loss_robust

                elif cfg.ADVT.NAME == 'mart':
                    criterion_kl = nn.KLDivLoss(reduction='none')
                    batch_size = len(data_batch['pc'])
                    data_batch['pc'] = Variable(data_batch['pc'], requires_grad=False)
                    loader_dataset_batch_proc = None
                    natural_inp = get_inp(cfg, task, model, x_natural, loader_dataset_batch_proc, dataset_name)
                    adv_inp = get_inp(cfg, task, model, data_batch, loader_dataset_batch_proc, dataset_name)
                    logits = model(**natural_inp)['logit']
                    logits_adv = model(**adv_inp)['logit']
                    y = data_batch['label'].to(logits.device)
                    adv_probs = F.softmax(logits_adv, dim=1)
                    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

                    nat_probs = F.softmax(logits, dim=1)

                    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

                    loss_robust = (1.0 / batch_size) * torch.sum(
                        torch.sum(criterion_kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (
                                    1.0000001 - true_probs))
                    loss = loss + 6.0 * loss_robust

                if loss.ne(loss).any():
                    print("WARNING: avoiding step as nan in the loss")
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    bad_grad = False
                    for x in model.parameters():
                        if x.grad is not None:
                            if x.grad.ne(x.grad).any():
                                print("WARNING: nan in a gradient")
                                bad_grad = True
                                break
                            if ((x.grad == float('inf')) | (x.grad == float('-inf'))).any():
                                print("WARNING: inf in a gradient")
                                bad_grad = True
                                break
                    if bad_grad:
                        print("WARNING: avoiding step as bad gradient")
                    else:
                        optimizer.step()

                    torch.cuda.empty_cache()

                time_data_loading += (time1 - time3)
                time_forward += (time2 - time1)
                time3 = time()
                time_backward += (time3 - time2)

                if i % 50 == 0:
                    print(
                        f"[{i}/{len(loader)}] avg_loss: {perf.agg_loss()}, FW time = {round(time_forward, 2)}, "
                        f"BW time = {round(time_backward, 2)}, DL time = {round(time_data_loading, 2)}")
        else:
            data_batch = {
                # 'pc': data_batch_ori[0],
                'pc': data_batch_ori[0][:, :, :3],
                'label': data_batch_ori[1].squeeze(),
                'index': data_batch_ori[2]
            }
            time1 = time()
            x_natural = data_batch.copy()

            if cfg.AUG.NAME == 'cutmix_r':
                data_batch = aug_utils.cutmix_r(data_batch,cfg)
            elif cfg.AUG.NAME == 'cutmix_k':
                data_batch = aug_utils.cutmix_k(data_batch,cfg)
            elif cfg.AUG.NAME == 'mixup':
                data_batch = aug_utils.mixup(data_batch,cfg)
            elif cfg.AUG.NAME == 'rsmix':
                data_batch = aug_utils.rsmix(data_batch,cfg)
            elif cfg.AUG.NAME == 'pgd':
                clip_func = None
                data_batch = aug_utils.pgd(cfg, clip_func, data_batch,model, task, loss_name, dataset_name)
                model.train()
            index = data_batch['index']
            loader_dataset_batch_proc = None
            inp = get_inp(cfg, task, model, data_batch, loader_dataset_batch_proc, dataset_name)

            out = model(inp['pc'].cuda() + poison[index])

            loss = get_loss(cfg, task, loss_name, data_batch, out, dataset_name, label_bias=label_bias, poison=poison)

            if cfg.EXP_EXTRA.get_fc_loss:
                fc_loss = get_loss(cfg, task, 'fc', data_batch, out, dataset_name, label_bias=label_bias, poison=poison)
                fc_perf = PerfTrackTrain(task, extra_param=get_extra_param())
                fc_perf.update_all(data_batch=data_batch, out=out, loss=fc_loss)

            if cfg.EXP_EXTRA.get_ce_loss:
                ce_loss = get_loss(cfg, task, 'cross_entropy', data_batch, out, dataset_name, label_bias=label_bias, poison=poison)
                ce_perf = PerfTrackTrain(task, extra_param=get_extra_param())
                ce_perf.update_all(data_batch=data_batch, out=out, loss=ce_loss)

            perf.update_all(data_batch=data_batch, out=out, loss=loss)
            time2 = time()


            ## robust loss
            if cfg.ADVT.NAME == 'trades':
                criterion_kl = nn.KLDivLoss(size_average=False)
                batch_size = len(data_batch['pc'])
                data_batch['pc'] = Variable(data_batch['pc'], requires_grad=False)
                loader_dataset_batch_proc = None
                natural_inp = get_inp(cfg, task, model, x_natural, loader_dataset_batch_proc, dataset_name)
                adv_inp = get_inp(cfg, task, model, data_batch, loader_dataset_batch_proc, dataset_name)
                loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(**adv_inp)['logit'], dim=1),
                                                        F.softmax(model(**natural_inp)['logit'], dim=1))
                loss = loss + loss_robust

            elif cfg.ADVT.NAME == 'mart':
                criterion_kl = nn.KLDivLoss(reduction='none')
                batch_size = len(data_batch['pc'])
                data_batch['pc'] = Variable(data_batch['pc'], requires_grad=False)
                loader_dataset_batch_proc = None
                natural_inp = get_inp(cfg, task, model, x_natural, loader_dataset_batch_proc, dataset_name)
                adv_inp = get_inp(cfg, task, model, data_batch, loader_dataset_batch_proc, dataset_name)
                logits = model(**natural_inp)['logit']
                logits_adv = model(**adv_inp)['logit']
                y = data_batch['label'].to(logits.device)
                adv_probs = F.softmax(logits_adv, dim=1)
                tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

                nat_probs = F.softmax(logits, dim=1)

                true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

                loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(criterion_kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
                loss = loss + 6.0 * loss_robust

            if loss.ne(loss).any():
                print("WARNING: avoiding step as nan in the loss")
            else:
                optimizer.zero_grad()
                loss.backward()
                bad_grad = False
                for x in model.parameters():
                    if x.grad is not None:
                        if x.grad.ne(x.grad).any():
                            print("WARNING: nan in a gradient")
                            bad_grad = True
                            break
                        if ((x.grad == float('inf')) | (x.grad == float('-inf'))).any():
                            print("WARNING: inf in a gradient")
                            bad_grad = True
                            break

                if bad_grad:
                    print("WARNING: avoiding step as bad gradient")
                else:
                    optimizer.step()

                torch.cuda.empty_cache()

            time_data_loading += (time1 - time3)
            time_forward += (time2 - time1)
            time3 = time()
            time_backward += (time3 - time2)

            if i % 50 == 0:
                print(
                    f"[{i}/{len(loader)}] avg_loss: {perf.agg_loss()}, FW time = {round(time_forward, 2)}, "
                    f"BW time = {round(time_backward, 2)}, DL time = {round(time_data_loading, 2)}")


    model.requires_grad_(False)
    if cfg.POISON is not None:
        if cfg.POISON in ['EM']:
            from attack import ProjectInnerClipLinfDelta
            clip_func = ProjectInnerClipLinfDelta(budget=cmd_args.eps)
            assert cfg.AUG.NAME == 'poison_pgd'
            label_bias = False if cfg.POISON == 'EM' else True
            for i, data_batch_ori in enumerate(loader):
                if cfg.EXP.DATASET == 'medpt':
                    poison_batch_size = 25
                    choice = np.random.choice(len(data_batch_ori[0]), len(data_batch_ori[0]), replace=False)
                    index_full = choice.reshape(-1, poison_batch_size)
                    for j in range(len(index_full)):
                        index = index_full[j]
                        data_batch = {
                            'pc': data_batch_ori[0][index][:, :, :3],
                            'label': data_batch_ori[1][index].squeeze(),
                            'index': index
                        }
                        poison[index] = aug_utils.poison_pgd(cfg, clip_func, data_batch, poison, model, task, loss_name, dataset_name,
                        step=10, alpha=-0.015, eps=cmd_args.eps, class_wise=False, label_bias=label_bias)
                else:
                    data_batch = {
                        'pc': data_batch_ori[0][:, :, :3],
                        'label': data_batch_ori[1].squeeze(),
                        'index': data_batch_ori[2]
                    }
                    index = data_batch['index']
                    poison[index] = aug_utils.poison_pgd(cfg, clip_func, data_batch, poison, model, task, loss_name,
                                                         dataset_name,
                                                         step=10, alpha=-0.015, eps=cmd_args.eps,
                                                         class_wise=False, label_bias=label_bias)


        elif cfg.POISON == 'REG_EM':
            from attack import CrossEntropyAdvLoss
            from attack import ChamferDist
            from attack import ProjectInnerClipLinf, ProjectInnerClipLinfDelta
            adv_func = CrossEntropyAdvLoss()
            dist_func = ChamferDist(method='standard')
            clip_func = ProjectInnerClipLinfDelta(budget=cmd_args.eps)
            for i, data_batch_ori in enumerate(loader):
                if cfg.EXP.DATASET == 'medpt':
                    poison_batch_size = 25
                    choice = np.random.choice(len(data_batch_ori[0]), len(data_batch_ori[0]), replace=False)
                    index_full = choice.reshape(-1, poison_batch_size)
                    for j in range(len(index_full)):
                        index = index_full[j]
                        data_batch = {
                            'pc': data_batch_ori[0][index][:, :, :3],
                            'label': data_batch_ori[1][index].squeeze(),
                            'index': index
                        }
                        poison[index], _ = aug_utils.chamf_ce_pgd(poison, data_batch, model, adv_func, dist_func, clip_func,
                                                step=10, attack_lr=0.015, class_wise=False, beta=cmd_args.chamf_coeff, errmin=True)
                else:
                    data_batch = {
                        'pc': data_batch_ori[0][:, :, :3],
                        'label': data_batch_ori[1].squeeze(),
                        'index': data_batch_ori[2]
                    }
                    index = data_batch['index']
                    poison[index], _ = aug_utils.chamf_ce_pgd(poison, data_batch, model, adv_func, dist_func,
                                                            clip_func,
                                                            step=10, attack_lr=0.015, class_wise=False,
                                                            beta=cmd_args.chamf_coeff, errmin=True)
        elif cfg.POISON in ['FC_EM']:
            from attack import CrossEntropyAdvLoss
            from attack import ChamferDist
            from attack import ProjectInnerClipLinf, ProjectInnerClipLinfDelta
            dist_func = ChamferDist(method='standard')
            clip_func = ProjectInnerClipLinfDelta(budget=cmd_args.eps)
            for i, data_batch_ori in enumerate(loader):
                if cfg.EXP.DATASET == 'medpt':
                    beta_coef = cmd_args.chamf_coeff * torch.ones(1600)
                    beta_coef[:100] *= 10
                    poison_batch_size = 25
                    choice = np.random.choice(len(data_batch_ori[0]), len(data_batch_ori[0]), replace=False)
                    index_full = choice.reshape(-1, poison_batch_size)
                    for j in range(len(index_full)):
                        index = index_full[j]
                        data_batch = {
                            'pc': data_batch_ori[0][index][:, :, :3],
                            'label': data_batch_ori[1][index].squeeze(),
                            'index': index
                        }
                        poison[index], beta_coef = aug_utils.chamf_fc_pgd(poison, data_batch, model, dist_func, clip_func,
                            step=cmd_args.steps, attack_lr=cmd_args.step_size, class_wise=False, label_bias=label_bias,
                            beta=cmd_args.chamf_coeff, temperature=cmd_args.temperature, num_classes=2)
                else:
                    data_batch = {
                        'pc': data_batch_ori[0][:, :, :3],
                        'label': data_batch_ori[1].squeeze(),
                        'index': data_batch_ori[2]
                    }
                    index = data_batch['index']
                    poison[index], beta_coef = aug_utils.chamf_fc_pgd(poison, data_batch, model, dist_func,
                clip_func, step=cmd_args.steps, attack_lr=cmd_args.step_size, class_wise=False, label_bias=label_bias,
                beta=cmd_args.chamf_coeff, temperature=cmd_args.temperature, num_classes=100)

        elif cfg.POISON == ['REG_EM']:
            from attack import CrossEntropyAdvLoss
            from attack import ChamferDist
            from attack import ProjectInnerClipLinf, ProjectInnerClipLinfDelta
            adv_func = CrossEntropyAdvLoss()
            dist_func = ChamferDist(method='standard')
            clip_func = ProjectInnerClipLinfDelta(budget=cmd_args.eps)
            for i, data_batch_ori in enumerate(loader):
                data_batch = {
                    'pc': data_batch_ori[0][:, :, :3],
                    'label': data_batch_ori[1].squeeze(),
                    'index': data_batch_ori[2]
                }
                index = data_batch['index']
                poison[index] = aug_utils.chamf_ce_pgd(poison, data_batch, model, adv_func, dist_func, clip_func,
                    label_bias=label_bias, step=10, attack_lr=0.015, class_wise=False, beta=cmd_args.chamf_coeff)

    model.requires_grad_(True)
    print(f" avg_loss: {perf.agg_loss()} ")
    if cfg.POISON is not None:
        return perf.agg(), perf.agg_loss(), poison, loss_stat, model
    else:
        return perf.agg(), perf.agg_loss(), loss_stat, model


def save_checkpoint(id, epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg):
    model.cpu()
    if cfg.POISON in ['FC_EM']:
        path = f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.temperature}_{cmd_args.eps}_{cmd_args.chamf_coeff}/model_{id}.pth"
    elif cfg.POISON in ['AP', 'AP_T', 'EM']:
        path = f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.eps}//model_{id}.pth"
    elif cfg.POISON in ['REG_EM', 'REG_AP', 'REG_AP_T']:
        path = f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.eps}_{cmd_args.chamf_coeff}/model_{id}.pth"
    else:
        path = f"./runs/{cfg.EXP.EXP_ID}/model_{id}.pth"
    torch.save({
        'cfg': vars(cfg),
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lr_sched_state': lr_sched.state_dict(),
        'bnm_sched_state': bnm_sched.state_dict() if bnm_sched is not None else None,
        'test_perf': test_perf,
    }, path)
    print('Checkpoint saved to %s' % path)
    model.to(DEVICE)


def load_best_checkpoint(model, cfg):
    path = f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.temperature}_{cmd_args.eps}_{cmd_args.chamf_coeff}/model_best.pth"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % path)


def load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path):
    print(f'Recovering model and checkpoint from {model_path}')
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state'])
            model = model.module

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if 'lr_sched_state' in checkpoint:
        lr_sched.load_state_dict(checkpoint['lr_sched_state'])
        if checkpoint['bnm_sched_state'] is not None:
            bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
    else:
        print("WARNING: lr scheduler and bnm scheduler states are not loaded.")

    return model


def get_model(cfg):
    if cfg.POISON in ['FC_EM'] or (cfg.EXP_EXTRA.get_feature is not None):
        get_feature = 'default'
    else:
        get_feature = False

    dropout = False if cfg.EXP_EXTRA.no_dropout else True
    print('dropout:', dropout)
    if cfg.EXP.MODEL_NAME == 'pointnet2':
        print('we use pointnet++!')
        model = models.PointNet2(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET, get_feature=get_feature,
            **cfg.MODEL.PN2)
    elif cfg.EXP.MODEL_NAME == 'dgcnn':
        print('we use dgcnn!')
        model = models.DGCNN(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET, get_feature=get_feature)
    elif cfg.EXP.MODEL_NAME == 'pointnet':
        print('we use pointnet!')
        model = models.PointNet(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET, get_feature=get_feature, dropout=dropout)
    else:
        assert False

    return model


def get_metric_from_perf(task, perf, metric_name):
    if task in ['cls', 'cls_trans']:
        assert metric_name in ['acc']
        metric = perf[metric_name]
    else:
        assert False
    return metric


def get_optimizer(optim_name, tr_arg, model):
    if optim_name == 'vanilla':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=tr_arg.lr_decay_factor,
            patience=tr_arg.lr_reduce_patience,
            verbose=True,
            min_lr=tr_arg.lr_clip)
        bnm_sched = None
    elif optim_name == 'pct':
        pass
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = lr_scheduler.CosineAnnealingLR(
            optimizer,
            tr_arg.num_epochs,
            eta_min=tr_arg.learning_rate)
        bnm_sched = None
    else:
        assert False

    return optimizer, lr_sched, bnm_sched


def entry_train(cfg, resume=False, model_path=""):

    if cfg.EXP.DATASET == 'medpt':
        from torch.utils.data import DataLoader

        from MedPT3D.intra3d import Intra3D
        num_points = 1024
        split = 4
        batch_size = 1600
        val_batch_size = 16
        test_batch_size = 16

        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        import random as rd
        torch.manual_seed(1)
        np.random.seed(1)
        rd.seed(1)

        if cfg.DATALOADER.poison_path is not None:
            try:
                if cfg.EXP.poison_type == 'FC_EM':
                    extra_path = f'{cmd_args.temperature}_{cmd_args.eps}_{cmd_args.chamf_coeff}'
                elif cfg.EXP.poison_type in ['REG_EM', 'REG_AP', 'REG_AP_T']:
                    extra_path = f'{cmd_args.eps}_{cmd_args.chamf_coeff}'
                elif cfg.EXP.poison_type in ['EM', 'AP', 'AP_T']:
                    extra_path = f'{cmd_args.eps}'
                else:
                    raise {'Method Not Support'}
                poison_path = f'{cfg.DATALOADER.poison_path}/{extra_path}/poison_{cfg.EXP.poison_type}.pt'
            except:
                poison_path = cfg.DATALOADER.poison_path
        else:
            poison_path = None
        train_loader = DataLoader(
            Intra3D(train_mode='train', cls_state=True, npoints=num_points, data_aug=False, choice=split,
            poison_path=poison_path),
            num_workers=0, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(
            Intra3D(train_mode='train', cls_state=True, npoints=num_points, data_aug=False, choice=split,
            poison_path=poison_path),
            num_workers=0, batch_size=val_batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(
            Intra3D(train_mode='test', cls_state=True, npoints=num_points, data_aug=False, choice=split),
            num_workers=0, batch_size=test_batch_size, shuffle=False, drop_last=False)
        loader_train = train_loader
        loader_valid = val_loader
        loader_test = test_loader
        sample_number = 1600
        num_points = 1024
        num_classes = 2

    elif cfg.EXP.DATASET == 'bsm2017':
        from torch.utils.data import DataLoader
        from face.data import BSM2017

        num_points = 1024
        batch_size = cfg.DATALOADER.batch_size
        split = 0.8
        num_classes = 100
        if cfg.DATALOADER.poison_path is not None:
            try:
                if cfg.EXP.poison_type == 'FC_EM':
                    extra_path = f'{cmd_args.temperature}_{cmd_args.eps}_{cmd_args.chamf_coeff}'
                elif cfg.EXP.poison_type in ['REG_EM', 'REG_AP', 'REG_AP_T']:
                    extra_path = f'{cmd_args.eps}_{cmd_args.chamf_coeff}'
                elif cfg.EXP.poison_type in ['EM', 'AP', 'AP_T']:
                    extra_path = f'{cmd_args.eps}'
                else:
                    raise {'Method Not Support'}
                poison_path = f'{cfg.DATALOADER.poison_path}/{extra_path}/poison_{cfg.EXP.poison_type}.pt'
            except:
                poison_path = cfg.DATALOADER.poison_path

            print('poison path:', poison_path)
        else:
            poison_path = None
        train_loader = DataLoader(
            BSM2017(train_mode='train',  npoints=num_points, data_aug=False, split=split,
            poison_path=poison_path, num_classes=num_classes),
            num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(
            BSM2017(train_mode='train', npoints=num_points, data_aug=False, split=split,
            poison_path=poison_path, num_classes=num_classes),
            num_workers=0, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(
            BSM2017(train_mode='test', npoints=num_points, data_aug=False, split=split, num_classes=num_classes),
            num_workers=0, batch_size=batch_size, shuffle=False, drop_last=False)

        loader_train = train_loader
        loader_valid = val_loader
        loader_test = test_loader
        sample_number = 4000
    else:
        raise {'dataset error, please use main.py'}
    
    model = get_model(cfg)
    model.to(DEVICE)
    print(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer, lr_sched, bnm_sched = get_optimizer(cfg.EXP.OPTIMIZER, cfg.TRAIN, model)

    loss_stat = []

    if resume:
        model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path)
    else:
        assert model_path == ""

    if cfg.POISON in ['FC_EM']:
        log_dir = f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.temperature}_{cmd_args.eps}_{cmd_args.chamf_coeff}"
    elif cfg.POISON in ['AP', 'AP_T', 'EM',]:
        log_dir = f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.eps}"
    elif cfg.POISON in ['REG_EM', 'REG_AP', 'REG_AP_T']:
        log_dir = f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.eps}_{cmd_args.chamf_coeff}"
    else:
        log_dir = f"./runs/{cfg.EXP.EXP_ID}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb = TensorboardManager(log_dir)
    track_train = TrackTrain(early_stop_patience=cfg.TRAIN.early_stop)

    poison = torch.zeros(sample_number, num_points, 3)
    poison = poison.cuda()

    for epoch in range(cfg.TRAIN.num_epochs):
        print(f'Epoch {epoch}')

        print('Training..')
        time_now = datetime.now()
        print('time:', time_now)
        if cfg.POISON is not None:
            train_perf, train_loss, poison, loss_stat, model = train(cfg.EXP.TASK, loader_train, model, optimizer,
                                           cfg.EXP.LOSS_NAME, cfg.EXP.DATASET, cfg, poison, loss_stat=loss_stat)
        else:
            train_perf, train_loss, loss_stat, model = train(cfg.EXP.TASK, loader_train, model, optimizer,
                                        cfg.EXP.LOSS_NAME, cfg.EXP.DATASET, cfg, poison, loss_stat=loss_stat)
        pprint.pprint(train_perf, width=80)
        tb.update('train', epoch, train_perf)

        if (not cfg.EXP_EXTRA.no_val) and epoch % cfg.EXP_EXTRA.val_eval_freq == 0:
            print('\nValidating..')
            val_perf = validate(cfg.EXP.TASK, loader_valid, model, cfg.EXP.DATASET)
            pprint.pprint(val_perf, width=80)
            tb.update('val', epoch, val_perf)
        else:
            val_perf = defaultdict(float)

        if (not cfg.EXP_EXTRA.no_test) and (epoch % cfg.EXP_EXTRA.test_eval_freq == 0):
            print('\nTesting..')
            test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET)
            pprint.pprint(test_perf, width=80)
            tb.update('test', epoch, test_perf)
        else:
            test_perf = defaultdict(float)

        track_train.record_epoch(
            epoch_id=epoch,
            train_metric=get_metric_from_perf(cfg.EXP.TASK, train_perf, cfg.EXP.METRIC),
            val_metric=get_metric_from_perf(cfg.EXP.TASK, val_perf, cfg.EXP.METRIC),
            test_metric=get_metric_from_perf(cfg.EXP.TASK, test_perf, cfg.EXP.METRIC))

        if (not cfg.EXP_EXTRA.no_val) and track_train.save_model(epoch_id=epoch, split='val'):
            print('Saving best model on the validation set')
            save_checkpoint('best_val', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if (not cfg.EXP_EXTRA.no_test) and track_train.save_model(epoch_id=epoch, split='test'):
            print('Saving best model on the test set')
            save_checkpoint('best_test', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if (not cfg.EXP_EXTRA.no_val) and track_train.early_stop(epoch_id=epoch):
            print(f"Early stopping at {epoch} as val acc did not improve for {cfg.TRAIN.early_stop} epochs.")
            break

        if (not (cfg.EXP_EXTRA.save_ckp == 0)) and (epoch % cfg.EXP_EXTRA.save_ckp == 0):
            save_checkpoint(f'{epoch}', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if cfg.EXP.OPTIMIZER == 'vanilla':
            assert bnm_sched is None
            lr_sched.step(train_loss)
        else:
            lr_sched.step()

        if cfg.POISON in ['FC_EM']:
            print('saving the poisons')
            torch.save(poison.cpu(), f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.temperature}_{cmd_args.eps}_{cmd_args.chamf_coeff}/poison_{cfg.POISON}.pt")
        elif cfg.POISON in ['EM']:
            print('saving the poisons')
            torch.save(poison.cpu(), f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.eps}/poison_{cfg.POISON}.pt")
        elif cfg.POISON in ['REG_EM']:
            print('saving the poisons')
            torch.save(poison.cpu(), f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.eps}_{cmd_args.chamf_coeff}/poison_{cfg.POISON}.pt")

    if not cmd_args.resume:
        print('Saving the final model')
        try:
            save_checkpoint('final', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)
            print('\nTesting on the final model..')
            last_test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET)
            pprint.pprint(last_test_perf, width=80)
        except:
            last_test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET)
            pprint.pprint(last_test_perf, width=80)


    time_now = datetime.now()
    print('time:', time_now)


    if cfg.POISON == 'AP':
        from attack import ProjectInnerClipLinfDelta
        clip_func = ProjectInnerClipLinfDelta(budget=cmd_args.eps)
        for i, data_batch_ori in enumerate(loader_train):
            if cfg.EXP.DATASET == 'medpt':
                poison_batch_size = 25
                choice = np.random.choice(len(data_batch_ori[0]), len(data_batch_ori[0]), replace=False)
                index_full = choice.reshape(-1, poison_batch_size)
                for j in range(len(index_full)):
                    index = index_full[j]
                    data_batch = {
                        'pc': data_batch_ori[0][index][:, :, :3],
                        'label': data_batch_ori[1][index].squeeze(),
                        'index': index
                    }
                    adv_data_batch = aug_utils.pgd(cfg, clip_func, data_batch, model,
                            cfg.EXP.TASK, cfg.EXP.LOSS_NAME, cfg.EXP.DATASET, alpha=0.001, step=250, eps=cmd_args.eps)
                    poison_batch = adv_data_batch['pc'] - data_batch['pc'].cuda()
                    try:
                        poison[index] = poison_batch
                    except:
                        poison = poison.to(DEVICE)
                        poison[index] = poison_batch
            else:
                data_batch = {
                    'pc': data_batch_ori[0][:, :, :3],
                    'label': data_batch_ori[1].squeeze(),
                    'index': data_batch_ori[2]
                }
                index = data_batch['index']

                adv_data_batch = aug_utils.pgd(cfg, clip_func, data_batch, model,
                                               cfg.EXP.TASK, cfg.EXP.LOSS_NAME, cfg.EXP.DATASET, alpha=0.001, step=250,
                                               eps=cmd_args.eps)
                poison_batch = adv_data_batch['pc'] - data_batch['pc'].cuda()
                try:
                    poison[index] = poison_batch
                except:
                    poison = poison.to(DEVICE)
                    poison[index] = poison_batch

    elif cfg.POISON == 'AP_T':
        from attack import ProjectInnerClipLinfDelta
        clip_func = ProjectInnerClipLinfDelta(budget=cmd_args.eps)
        for i, data_batch_ori in enumerate(loader_train):
            if cfg.EXP.DATASET == 'medpt':
                poison_batch_size = 25
                choice = np.random.choice(len(data_batch_ori[0]), len(data_batch_ori[0]), replace=False)
                index_full = choice.reshape(-1, poison_batch_size)
                for j in range(len(index_full)):
                    index = index_full[j]
                    data_batch = {
                        'pc': data_batch_ori[0][index][:, :, :3],
                        'label': data_batch_ori[1][index].squeeze(),
                        'index': index
                    }

                    adv_data_batch = aug_utils.pgd(cfg, clip_func, data_batch, model, cfg.EXP.TASK, cfg.EXP.LOSS_NAME, cfg.EXP.DATASET,
                                                   alpha=-0.001,  step=250, eps=cmd_args.eps, label_bias=True, num_classes=num_classes)
                    poison_batch = adv_data_batch['pc'] - data_batch['pc'].cuda()
                    try:
                        poison[index] = poison_batch
                    except:
                        poison = poison.to(DEVICE)
                        poison[index] = poison_batch
            else:
                data_batch = {
                    'pc': data_batch_ori[0][:, :, :3],
                    'label': data_batch_ori[1].squeeze(),
                    'index': data_batch_ori[2]
                }
                index = data_batch['index']

                adv_data_batch = aug_utils.pgd(cfg, clip_func, data_batch, model, cfg.EXP.TASK, cfg.EXP.LOSS_NAME, cfg.EXP.DATASET,
                                               alpha=-0.001, step=250, eps=cmd_args.eps, label_bias=True, num_classes=num_classes)
                poison_batch = adv_data_batch['pc'] - data_batch['pc'].cuda()
                try:
                    poison[index] = poison_batch
                except:
                    poison = poison.to(DEVICE)
                    poison[index] = poison_batch

    elif cfg.POISON == 'REG_AP':
        from attack import CrossEntropyAdvLoss
        from attack import ChamferDist
        from attack import ProjectInnerClipLinf,ProjectInnerClipLinfDelta
        adv_func = CrossEntropyAdvLoss()
        dist_func = ChamferDist(method='standard')
        clip_func = ProjectInnerClipLinfDelta(budget=cmd_args.eps)

        for i, data_batch_ori in enumerate(loader_train):
            if cfg.EXP.DATASET == 'medpt':
                poison_batch_size = 25
                choice = np.random.choice(len(data_batch_ori[0]), len(data_batch_ori[0]), replace=False)
                index_full = choice.reshape(-1, poison_batch_size)
                for j in range(len(index_full)):
                    index = index_full[j]
                    data_batch = {
                        'pc': data_batch_ori[0][index][:, :, :3],
                        'label': data_batch_ori[1][index].squeeze(),
                        'index': index
                    }
                    delta = torch.zeros(sample_number, num_points, 3)
                    poison[index], _ = aug_utils.chamf_ce_pgd(delta, data_batch, model, adv_func, dist_func, clip_func,
                                                    step=250, attack_lr=0.001, beta=cmd_args.chamf_coeff)
            else:
                data_batch = {
                    'pc': data_batch_ori[0][:, :, :3],
                    'label': data_batch_ori[1].squeeze(),
                    'index': data_batch_ori[2]
                }
                index = data_batch['index']
                delta = torch.zeros(sample_number, num_points, 3)
                poison[index], _ = aug_utils.chamf_ce_pgd(delta, data_batch, model, adv_func, dist_func, clip_func,
                                                        step=250, attack_lr=0.001, beta=cmd_args.chamf_coeff)


    elif cfg.POISON == 'REG_AP_T':
        from attack import CrossEntropyAdvLoss
        from attack import ChamferDist
        from attack import ProjectInnerClipLinf,ProjectInnerClipLinfDelta
        adv_func = CrossEntropyAdvLoss()
        dist_func = ChamferDist(method='standard')
        clip_func = ProjectInnerClipLinfDelta(budget=cmd_args.eps)

        for i, data_batch_ori in enumerate(loader_train):
            if cfg.EXP.DATASET == 'medpt':
                poison_batch_size = 25
                choice = np.random.choice(len(data_batch_ori[0]), len(data_batch_ori[0]), replace=False)
                index_full = choice.reshape(-1, poison_batch_size)
                for j in range(len(index_full)):
                    index = index_full[j]
                    data_batch = {
                        'pc': data_batch_ori[0][index][:, :, :3],
                        'label': data_batch_ori[1][index].squeeze(),
                        'index': index
                    }
                    delta = torch.zeros(sample_number, num_points, 3)
                    poison[index], _ = aug_utils.chamf_ce_pgd(delta, data_batch, model, adv_func, dist_func, clip_func,
                                            step=250, attack_lr=0.001, label_bias=True, beta=cmd_args.chamf_coeff, num_classes=num_classes)
            else:
                data_batch = {
                    'pc': data_batch_ori[0][:, :, :3],
                    'label': data_batch_ori[1].squeeze(),
                    'index': data_batch_ori[2]
                }
                index = data_batch['index']
                delta = torch.zeros(sample_number, num_points, 3)
                poison[index], _ = aug_utils.chamf_ce_pgd(delta, data_batch, model, adv_func, dist_func, clip_func,
                                                        step=250, attack_lr=0.001, label_bias=True,
                                                        beta=cmd_args.chamf_coeff, num_classes=num_classes)

    if cfg.POISON in ['FC_EM']:
        print('saving the poisons')
        torch.save(poison.cpu(), f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.temperature}_{cmd_args.eps}_{cmd_args.chamf_coeff}/poison_{cfg.POISON}.pt")
    elif cfg.POISON in ['AP', 'AP_T', 'EM']:
        print('saving the poisons')
        torch.save(poison.cpu(), f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.eps}/poison_{cfg.POISON}.pt")
    elif cfg.POISON in ['REG_EM', 'REG_EM_T', 'REG_AP', 'REG_AP_T']:
        print('saving the poisons')
        torch.save(poison.cpu(), f"./runs/{cfg.EXP.EXP_ID}/{cmd_args.eps}_{cmd_args.chamf_coeff}/poison_{cfg.POISON}.pt")

    time_now = datetime.now()
    print('time:', time_now)

    tb.close()

if __name__ == '__main__':

    assert not cmd_args.exp_config == ""
    if not cmd_args.resume:
        assert cmd_args.model_path == ""

    cfg = get_cfg_defaults()
    cfg.merge_from_file(cmd_args.exp_config)
    if cfg.EXP.EXP_ID == "":
        cfg.EXP.EXP_ID = str(datetime.now())[:-7].replace(' ', '-')
    cfg.freeze()
    print(cfg)

    random.seed(cfg.EXP.SEED)
    np.random.seed(cfg.EXP.SEED)
    torch.manual_seed(cfg.EXP.SEED)

    entry_train(cfg, cmd_args.resume, cmd_args.model_path)
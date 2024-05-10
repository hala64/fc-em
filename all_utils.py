import tensorboardX
import torch
import torch.nn.functional as F
import numpy as np


DATASET_NUM_CLASS = {
    'modelnet40_dgcnn': 40,
    'scanobjectnn': 15,
    'medpt': 2,
    'bsm2017': 100
}

class TensorboardManager:
    def __init__(self, path):
        self.writer = tensorboardX.SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            self.writer.add_scalar('%s_%s' % (split, k), v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class TrackTrain:
    def __init__(self, early_stop_patience):
        self.early_stop_patience = early_stop_patience
        self.counter = -1
        self.best_epoch_val = -1
        self.best_epoch_train = -1
        self.best_epoch_test = -1
        self.best_val = float("-inf")
        self.best_test = float("-inf")
        self.best_train = float("-inf")
        self.test_best_val = float("-inf")

    def record_epoch(self, epoch_id, train_metric, val_metric, test_metric):
        assert epoch_id == (self.counter + 1)
        self.counter += 1

        if val_metric >= self.best_val:
            self.best_val = val_metric
            self.best_epoch_val = epoch_id
            self.test_best_val = test_metric

        if test_metric >= self.best_test:
            self.best_test = test_metric
            self.best_epoch_test = epoch_id

        if train_metric >= self.best_train:
            self.best_train = train_metric
            self.best_epoch_train = epoch_id


    def save_model(self, epoch_id, split):
        assert epoch_id == self.counter
        if split == 'val':
            if self.best_epoch_val == epoch_id:
                _save_model = True
            else:
                _save_model = False
        elif split == 'test':
            if self.best_epoch_test == epoch_id:
                _save_model = True
            else:
                _save_model = False
        elif split == 'train':
            if self.best_epoch_train == epoch_id:
                _save_model = True
            else:
                _save_model = False
        else:
            assert False

        return _save_model

    def early_stop(self, epoch_id):
        assert epoch_id == self.counter
        if (epoch_id - self.best_epoch_val) > self.early_stop_patience:
            return True
        else:
            return False


class PerfTrackVal:
    def __init__(self, task, extra_param=None):
        self.task = task
        if task in ['cls', 'cls_trans']:
            assert extra_param is None
            self.all = []
            self.class_seen = None
            self.class_corr = None
        else:
            assert False
    def update(self, data_batch, out):
        if self.task in ['cls', 'cls_trans']:
            correct = self.get_correct_list(out['logit'], data_batch['label'])
            self.all.extend(correct)
            self.update_class_see_corr(out['logit'], data_batch['label'])
        else:
            assert False
    def agg(self):
        if self.task in ['cls', 'cls_trans']:
            perf = {
                'acc': self.get_avg_list(self.all),
                'class_acc': np.mean(np.array(self.class_corr) / np.array(self.class_seen,dtype=np.float32))
            }
        else:
            assert False
        return perf

    def update_class_see_corr(self, logit, label):
        if self.class_seen is None:
            num_class = logit.shape[1]
            self.class_seen = [0] * num_class
            self.class_corr = [0] * num_class

        pred_label = logit.argmax(axis=1).to('cpu').tolist()
        for _pred_label, _label in zip(pred_label, label):
            self.class_seen[_label] += 1
            if _pred_label == _label:
                self.class_corr[_pred_label] += 1

    @staticmethod
    def get_correct_list(logit, label):
        label = label.to(logit.device)
        pred_class = logit.argmax(axis=1)
        return (label == pred_class).to('cpu').tolist()
    @staticmethod
    def get_avg_list(all_list):
        for x in all_list:
            assert isinstance(x, bool)
        return sum(all_list) / len(all_list)


class PerfTrackTrain(PerfTrackVal):
    def __init__(self, task, extra_param=None):
        super().__init__(task, extra_param)
        self.all_loss = []

    def update_loss(self, loss):
        self.all_loss.append(loss.item())

    def agg_loss(self):
        return sum(self.all_loss) / len(self.all_loss)

    def update_all(self, data_batch, out, loss):
        self.update(data_batch, out)
        self.update_loss(loss)

def smooth_loss(pred, gold):
    eps = 0.2

    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()

    return loss

def chamfer_loss(data_batch, poison, class_wise=False):
    from attack import ChamferDist
    dist_func = ChamferDist(method='adv2ori')

    data = data_batch['pc'].cuda().detach()
    target = data_batch['label'].cuda().detach()
    index = data_batch['index']

    ori_data = data.clone().detach()

    B, K = data.shape[:2]

    if class_wise:
        adv_data = data + poison[target]
    else:
        adv_data = data + poison[index]
    dist_loss = dist_func(
        adv_data.contiguous(),
        ori_data.contiguous()).mean() * K

    return dist_loss

def chamfer_loss_standard(data_batch, poison, class_wise=False):
    from attack import ChamferDist
    dist_func = ChamferDist(method='standard')

    data = data_batch['pc'].cuda().detach()
    target = data_batch['label'].cuda().detach()
    index = data_batch['index']

    ori_data = data.clone().detach()

    B, K = data.shape[:2]

    if class_wise:
        adv_data = data + poison[target]
    else:
        adv_data = data + poison[index]
    dist_loss = dist_func(
        adv_data.contiguous(),
        ori_data.contiguous()).mean() * K

    return dist_loss


def knn_loss(data_batch, poison, class_wise=False):
    from attack import ChamferkNNDist
    dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                               knn_k=5, knn_alpha=1.05,
                               chamfer_weight=5., knn_weight=3.)

    data = data_batch['pc'].cuda().detach()
    target = data_batch['label'].cuda().detach()
    index = data_batch['index']

    ori_data = data.clone().detach()

    B, K = data.shape[:2]

    if class_wise:
        adv_data = data + poison[target]
    else:
        adv_data = data + poison[index]
    dist_loss = dist_func(
        adv_data.contiguous(),
        ori_data.contiguous()).mean() * K

    return dist_loss

# based on: https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_classification.py
import torch.nn as nn
from pointnet_pyt.pointnet.model import PointNetCls
from all_utils import DATASET_NUM_CLASS

class PointNet(nn.Module):

    def __init__(self, dataset, task, get_feature=None, dropout=True):
        super().__init__()
        self.task = task
        self.get_feature = get_feature
        num_class = DATASET_NUM_CLASS[dataset]
        print(num_class)
        self.model = PointNetCls(k=num_class, feature_transform=True, get_feature=get_feature, dropout=dropout)

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.transpose(2, 1).float()

        if self.get_feature is not None:
             logit, _, trans_feat, feature = self.model(pc)
             out = {'logit': logit, 'trans_feat': trans_feat, 'feature': feature}
        else:
            logit, _, trans_feat = self.model(pc)
            out = {'logit': logit, 'trans_feat': trans_feat}
        return out

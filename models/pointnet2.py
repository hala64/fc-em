import torch
import torch.nn as nn
from pointnet2_pyt.pointnet2.models.pointnet2_msg_cls import Pointnet2MSG
from all_utils import DATASET_NUM_CLASS

class PointNet2(nn.Module):

    def __init__(self, task, dataset, get_feature, version_cls):
        super().__init__()
        self.task =  task
        self.get_feature = get_feature
        num_class = DATASET_NUM_CLASS[dataset]
        if task == 'cls':
            self.model = Pointnet2MSG(num_classes=num_class, input_channels=0, use_xyz=True,
                                      version=version_cls, get_feature=get_feature)
        else:
            assert False

    def forward(self, pc, normal=None, cls=None):
        #pc = pc.to(next(self.parameters()).device)
        pc = pc.cuda()
        if self.task == 'cls':
            assert cls is None
            assert normal is None
            if self.get_feature is not None:
                logit, feature = self.model(pc)
                out = {'logit': logit, 'feature': feature}
            else:
                logit = self.model(pc)
                out = {'logit': logit}
        else:
            assert False
        return out


import torch.nn as nn
import torch.nn.functional as F
from dgcnn.pytorch.model import DGCNN as DGCNN_original
from all_utils import DATASET_NUM_CLASS

class DGCNN(nn.Module):

    def __init__(self, task, dataset, get_feature=None):
        super().__init__()
        self.task = task
        self.dataset = dataset
        self.get_feature = get_feature

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.k = 20
                    self.emb_dims = 1024
                    self.dropout = 0.5
                    self.leaky_relu = 1
            args = Args()
            self.model = DGCNN_original(args, output_channels=num_classes, get_feature=get_feature)

        else:
            assert False

    def forward(self, pc, cls=None):
        #pc = pc.to(next(self.parameters()).device)
        pc = pc.cuda()
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            if self.get_feature is not None:
                logit, feature = self.model(pc)
                out = {'logit': logit,
                       'feature': feature}
            else:
                logit = self.model(pc)
                out = {'logit': logit}
        else:
            assert False

        return out

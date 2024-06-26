"""Implementation of Untargeted AdvPC CW Attack for point perturbation.
"""

import pdb
import time
import torch
import torch.optim as optim
import numpy as np

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=0.5, scale_high=1.5, translate_range=0.0, no_z_aug=False):
        """
        :param scale_low:
        :param scale_high:
        :param translate_range:
        :param no_z: no translation and scaling along the z axis
        """
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range
        self.no_z_aug = no_z_aug

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            # xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            # if self.no_z_aug:
            #     xyz1[2] = 1.0
            #     xyz2[2] = 0.0
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) #+ torch.from_numpy(xyz2).float().cuda()

        return pc

# def shear(pointcloud,severity=1):
#     pointcloud = pointcloud.transpose(1, 2)
#     B, N, C = pointcloud.shape
#     c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
#     a = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
#     b = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
#     d = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
#     e = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
#     f = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
#     g = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])

#     matrix = torch.from_numpy(np.array([[1,0,b],[d,1,e],[f,0,1]])).view(1,3,3).expand(B,-1,-1).cuda().float()
#     new_pc = torch.matmul(pointcloud,matrix) #.astype('float32')
#     new_pc = new_pc.transpose(2, 1)
#     return new_pc

def shear(pointcloud,severity=1):
    pointcloud = pointcloud.transpose(1, 2)
    B, N, C = pointcloud.shape
    # c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
    cl = 0.05
    ch = 0.10
    a = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])
    b = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])
    d = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])
    e = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])
    f = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])
    g = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])

    matrix = torch.from_numpy(np.array([[1,0,b],[d,1,e],[f,0,1]])).view(1,3,3).expand(B,-1,-1).cuda().float()
    new_pc = torch.matmul(pointcloud,matrix) #.astype('float32')
    new_pc = new_pc.transpose(2, 1)
    return new_pc


class CWUAdvPC:
    """Class for CW UAdvPC attack.
    """

    def __init__(self, model, ae_model, adv_func, dist_func, attack_lr=1e-2,
                 binary_step=2, num_iter=200, GAMMA=0.5, clip_func=None, pa = 0.5, ps=0.5):
        """CW attack by perturbing points.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            dist_func (function): distance metric
            attack_lr (float, optional): lr for optimization. Defaults to 1e-2.
            binary_step (int, optional): binary search step. Defaults to 10.
            num_iter (int, optional): max iter num in every search step. Defaults to 500.
        """

        self.model = model.cuda()
        self.model.eval()
        self.ae_model = ae_model.cuda()
        self.ae_model.eval()

        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.GAMMA = GAMMA
        self.clip_func = clip_func
        self.pa = pa
        self.ps = ps

    def attack(self, data, target):
        """Attack on given data to target.

        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        data = data.transpose(1, 2).contiguous()
        ori_data = data.clone().detach()
        ori_data.requires_grad = False
        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()  # [B]

        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))

        # perform binary search
        for binary_step in range(self.binary_step):
            # init variables with small perturbation
            adv_data = ori_data.clone().detach() + \
                torch.randn((B, 3, K)).cuda() * 1e-7
            adv_data.requires_grad_()
            opt = optim.Adam([adv_data], lr=self.attack_lr, weight_decay=0.)

            adv_loss = torch.tensor(0.).cuda()
            dist_loss = torch.tensor(0.).cuda()

            total_time = 0.
            optimize_time = 0.
            clip_time = 0.
            update_time = 0.

            # one step in binary search
            for iteration in range(self.num_iter):
                t1 = time.time()

                #original adversarial loss
                # r = np.random.rand(1)
                # if r <= self.pa:
                #     ## shear
                #     r1 = np.random.rand(1)
                #     if r1 <= self.ps:
                #         nadv_data = shear(adv_data)
                #         # nadv_data = cut_points_knn(adv_data)
                #         # nadv_data = adv_data + torch.randn((B, 3, K)).cuda() * 1e-4
                #         logits = self.model(nadv_data)
                #     else:
                #         scale = PointcloudScaleAndTranslate()
                #         # adv_data.data = scale(adv_data.data)
                #         nadv_data = scale(adv_data.data)
                #         logits = self.model(nadv_data)
                # else:
                #     logits = self.model(adv_data)
                logits = self.model(**{'pc':adv_data.transpose(1, 2)})['logit']  # [B, num_classes]
                if isinstance(logits, tuple):  # PointNet
                    logits = logits[0]

                adv_loss = (1-self.GAMMA) * self.adv_func(logits, target).mean()
                opt.zero_grad()
                adv_loss.backward()

                #autoencoder adversarial loss
                adv_data_constr = self.ae_model(adv_data)
                ae_logits = self.model(**{'pc':adv_data_constr.transpose(1, 2)})['logit']
                if isinstance(ae_logits, tuple):  # PointNet
                    ae_logits = ae_logits[0]

                ae_adv_loss = self.GAMMA * self.adv_func(ae_logits, target).mean()
                ae_adv_loss.backward()
                opt.step()

                t2 = time.time()
                optimize_time += t2 - t1

                #clip
                adv_data.data = self.clip_func(adv_data.detach().clone(), ori_data)

                t3 = time.time()
                clip_time += t3 - t2

                # print
                pred = torch.argmax(logits, dim=1)  # [B]
                ae_pred = torch.argmax(ae_logits, dim=1)  # [B]
                success_num = ((pred != target)*(ae_pred != target)).sum().item()
                if iteration % (self.num_iter // 5) == 0:
                    print('Step {}, iteration {}, success {}/{}\n'
                          'adv_loss: {:.4f}, dist_loss: {:.4f}'.
                          format(binary_step, iteration, success_num, B,
                                 (adv_loss.item()+ae_adv_loss.item()), dist_loss.item()))

                # record values!
                dist_val = torch.sqrt(torch.sum(
                    (adv_data - ori_data) ** 2, dim=[1, 2])).\
                    detach().cpu().numpy()  # [B]
                pred_val = pred.detach().cpu().numpy()  # [B]
                ae_pred_val = ae_pred.detach().cpu().numpy()  # [B]
                input_val = adv_data.detach().cpu().numpy()  # [B, 3, K]

                # update
                for e, (dist, pred, ae_pred, label, ii) in \
                        enumerate(zip(dist_val, pred_val, ae_pred_val, label_val, input_val)):
                    if dist < o_bestdist[e] and pred != label and (ae_pred != label or self.GAMMA < 0.001):
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii

                t4 = time.time()
                update_time += t4 - t3

                total_time += t4 - t1

                if iteration % 100 == 0:
                    print('total time: {:.2f}, opt: {:.2f}, '
                          'clip: {:.2f}, update: {:.2f}'.
                          format(total_time, optimize_time,
                                 clip_time, update_time))
                    total_time = 0.
                    optimize_time = 0.
                    clip_time = 0.
                    update_time = 0.
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

        # end of CW attack
        # fail to attack some examples
        fail_idx = (o_bestscore < 0)
        o_bestattack[fail_idx] = input_val[fail_idx]

        adv_pc = torch.tensor(o_bestattack).to(adv_data)
        adv_pc = self.clip_func(adv_pc, ori_data)

        logits = self.model(**{'pc':adv_data.transpose(1, 2)})['logit']
        if isinstance(logits, tuple):  # PointNet
            logits = logits[0]
        preds = torch.argmax(logits, dim=-1)
        # return final results
        success_num = (preds != target).sum().item()
        print('Successfully attack {}/{}'.format(success_num, B))
        return o_bestdist, adv_pc.detach().cpu().numpy().transpose((0, 2, 1)), success_num

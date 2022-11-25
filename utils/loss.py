# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


def get_sigma(input, eps=1e-7):
    _, wh, theta = input.split([2, 2, 1], -1)
    wh = wh.clamp(min=eps)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    R = torch.cat((Cos, -Sin, Sin, Cos), -1).view(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    sigma = (R @ S.square() @ R.transpose(1, 2)).reshape(-1, 2, 2)
    return sigma


def compute_gwd(pred, target, eps=1e-7, alpha=1.0, tau=1.0, norm=True):
    pred_xy = pred[..., :2]
    target_xy = target[..., :2]
    pred_sigma = get_sigma(pred, eps)
    target_sigma = get_sigma(target, eps)
    # m calculate
    xy_dist = (pred_xy - target_xy).square().sum(-1)
    whr_dist = pred_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_dist = whr_dist + target_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_tr = (pred_sigma @ target_sigma).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (pred_sigma.det() * target_sigma.det()).clamp(0).sqrt()
    whr_dist = whr_dist + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()
    )
    dist = (xy_dist + alpha * alpha * whr_dist).clamp(0).sqrt()
    if norm:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(eps)
        dist = dist / scale
    loss = 1 - 1 / (tau + torch.log1p(dist))
    return loss


def compute_kld(pred, target, alpha=1.0, tau=1.0, sqrt=True, eps=1e-7):
    xy_p = pred[..., :2]
    xy_t = target[..., :2]
    Sigma_p = get_sigma(pred)
    Sigma_t = get_sigma(target)
    _shape = xy_p.shape
    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(
        dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(
        Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(0).sqrt()

    distance = distance.reshape(_shape[:-1])
    loss = 1 - 1 / (tau + torch.log1p(distance))

    return loss


class KLDLoss(nn.Module):
    def __init__(self,
                 taf=1.0,
                 eps=1e-6):
        super().__init__()
        self.eps = eps
        self.taf = taf

    def forward(self,
                pred,
                target,
                avg_factor=1.,
                **kwargs):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 5)
        target = target.view(-1, 5)

        delta_x = pred[:, 0] - target[:, 0]
        delta_y = pred[:, 1] - target[:, 1]
        # 角度制-弧度制
        # pre_angle_radian = 3.141592653589793 * pred[:, 4] / 180.0
        pre_angle_radian = pred[:, 4]
        # targrt_angle_radian = 3.141592653589793 * target[:, 4] / 180.0
        targrt_angle_radian = target[:, 4]
        delta_angle_radian = pre_angle_radian - targrt_angle_radian

        kld = 0.5 * (
                4 * torch.pow(
            (delta_x.mul(torch.cos(targrt_angle_radian)) + delta_y.mul(torch.sin(targrt_angle_radian))),
            2) / torch.pow(target[:, 2], 2)
                + 4 * torch.pow(
            (delta_y.mul(torch.cos(targrt_angle_radian)) - delta_x.mul(torch.sin(targrt_angle_radian))),
            2) / torch.pow(target[:, 3], 2)
        ) \
              + 0.5 * (
                      torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(
                  torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(
                  torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(
                  torch.cos(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(
                  torch.cos(delta_angle_radian), 2)
              ) \
              + 0.5 * (
                      torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                      + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
              ) \
              - 1.0

        kld_loss = 1 - 1 / (self.taf + torch.log(kld + 1))

#         if self.reduction == "mean":
#             loss = kld_loss.mean()
#         elif self.reduction == "sum":
#             loss = kld_loss.sum()
#         else:
#             raise NotImplemented
        return kld_loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.kld_loss = KLDLoss()  # 导入kld_loss损失

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.stride = det.stride  # tensor([8., 16., 32., ...])
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels])

        Return：
            total_loss * bs (tensor): [1] 
            torch.cat((lbox, lobj, lcls, ltheta)).detach(): [4]
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets, (n_targets, self.no)

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # featuremap pixel
                class_index = 5 + self.nc
                theat = (ps[:, class_index:].sigmoid() - 0.5) * 3.1415926
                pbox = torch.cat((pxy, pwh, theat), 1)  # predicted box
                iou = self.kld_loss(pbox, tbox[i])  # iou(prediction, target)
                lbox += iou.mean()  # iou loss

                # Objectness
                score_iou = (1 - iou).detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # 替换原有的使用iou来作为obj的置信度, 将所有包含物体的先验框置信度设置为1

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t = torch.full_like(ps[:, 5:class_index], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    lcls += self.BCEcls(ps[:, 5:class_index], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        # return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) pixel

        Return：non-normalized data
            tcls (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
            tbox (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 4) featuremap pixel
            indices (list[P3_out,...]): len=self.na, tensor.size(4, n_filter2) [b, a, gj, gi]
            anch (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 2)
            tgaussian_theta (list[P3_out,...]): len=self.na, tensor.size(n_filter2, hyp['cls_theta'])
            # ttheta (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
        """
        # Build targets for compute_loss()
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # ttheta, tgaussian_theta = [], []
        tgaussian_theta = []
        # gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        feature_wh = torch.ones(2, device=targets.device)  # feature_wh
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets (tensor): (n_gt_all_batch, c) -> (na, n_gt_all_batch, c) -> (na, n_gt_all_batch, c+1)
        # targets (tensor): (na, n_gt_all_batch, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index]])
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],  # tensor: (5, 2)
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            # gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain=[1, 1, w, h, w, h, 1, 1]
            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]  # xyxy gain=[w_f, h_f]

            # Match targets to anchors
            # t = targets * gain # xywh featuremap pixel
            t = targets.clone()  # (na, n_gt_all_batch, c+1)
            t[:, :, 2:6] /= self.stride[i]  # xyls featuremap pixel
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # edge_ls ratio, torch.size(na, n_gt_all_batch, 2)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare, torch.size(na, n_gt_all_batch)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter; Tensor.size(n_filter1, c+1)

                # Offsets
                gxy = t[:, 2:4]  # grid xy; (n_filter1, 2)
                # gxi = gain[[2, 3]] - gxy  # inverse
                gxi = feature_wh[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # (5, n_filter1)
                t = t.repeat((5, 1, 1))[j]  # (n_filter1, c+1) -> (5, n_filter1, c+1) -> (n_filter2, c+1)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # (5, n_filter1, 2) -> (n_filter2, 2)
            else:
                t = targets[0]  # (n_gt_all_batch, c+1)
                offsets = 0

            # Define, t (tensor): (n_filter2, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index])
            b, c = t[:, :2].long().T  # image, class; (n_filter2)
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            theta = t[:, 6:7]  # 取出真实框的角度
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices 取整
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            indices.append(
                (b, a, gj.clamp_(0, feature_wh[1] - 1), gi.clamp_(0, feature_wh[0] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh, theta), 1))  # box[x, y, w, h ,theta]
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # ttheta.append(theta) # theta, θ∈[-pi/2, pi/2)

        # return tcls, tbox, indices, anch
        return tcls, tbox, indices, anch  # , ttheta

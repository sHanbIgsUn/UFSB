# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch
from module.kernels import GaussianKernel
from .adv_loss import DomainAdversarialLoss
from .jmmd_loss import JointMultipleKernelMaximumMeanDiscrepancy
from .ot_loss import OptimalTransport
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes,domainDiscriminator,sigmoid=False): 
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'adv' in cfg.MODEL.MODULE:
        adv_loss=DomainAdversarialLoss(domainDiscriminator,sigmoid=sigmoid)
        print("using adv loss for training")

    if 'jmmd' in cfg.MODEL.MODULE:
        jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=(
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                (GaussianKernel(sigma=0.92, track_running_stats=False),)
            ),
            linear=False, thetas=None
        )
        print("using jmmd loss for training")

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        # xent_r = CrossEntropyLabelSmooth(num_classes=cfg.DATASETS.NUM_REAL_ID)
        # xent_u = CrossEntropyLabelSmooth(num_classes=num_classes - cfg.DATASETS.NUM_REAL_ID)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore = None,writer=None,iter=0):
            if 'ms1k_cufsf_fake' in cfg.DATASETS.NAMES or 'ms1k_iiit_fake' in cfg.DATASETS.NAMES:
                sketch = (
                    (target_cam == 6) | (target_cam == 7) |
                    (target_cam == 8) | (target_cam == 9) |
                    (target_cam == 10) | (target_cam == 11) |
                    (target_cam == 14) | (target_cam == 15) |
                    (target_cam == 16) | (target_cam == 18) 
                )
            # if 'face' in cfg.DATASETS.NAMES:
            elif 'ms1k_cufsf' == cfg.DATASETS.NAMES:
                sketch = (
                    (target_cam == 6) | (target_cam == 7) |
                    (target_cam == 8) | (target_cam == 9) |
                    (target_cam == 10) | (target_cam == 11) |
                    (target_cam == 12) 
                )
            elif 'ms1k' in cfg.DATASETS.NAMES:
                sketch = (
                    (target_cam == 6) | (target_cam == 7) |
                    (target_cam == 8) | (target_cam == 9) |
                    (target_cam == 10) | (target_cam == 11) |
                    (target_cam == 14) | (target_cam == 15)
                )
            else:
                sketch = (
                    (target_cam == 0) | (target_cam == 2)  )
            # mask_real = target < 
            # target_r = target[mask_real]
            # target_u = target[~mask_real] - cfg.DATASETS.NUM_REAL_ID[0]
            # sketch_real = sketch & mask_real
            # sketch_unreal = sketch & ~mask_real

            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        # ID_LOSS_R = [xent_r(scor, target_r) for scor in score[0:2]]
                        # ID_LOSS_R = sum(ID_LOSS_R)
                        # ID_LOSS_U = [xent_u(scor, target_u) for scor in score[2:4]]
                        # ID_LOSS_U = sum(ID_LOSS_U)
                        # ID_LOSS = ID_LOSS_R*0.7 + ID_LOSS_U*0.3
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)
                    writer.add_scalar("train/ID_LOSS", ID_LOSS.item(), iter)
                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, target)[0]
                    writer.add_scalar("train/TRI_LOSS", TRI_LOSS.item(), iter)
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    if i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                        writer.add_scalar("train/I2TLOSS", I2TLOSS.item(), iter)


                    if 'adv' in cfg.MODEL.MODULE:
                        if isinstance(feat, list):
                            ADV_LOSS = [adv_loss(feats[sketch_unreal], feats[sketch_real]) for feats in feat[0:]]
                            ADV_LOSS = sum(ADV_LOSS)
                            domain_acc = adv_loss.domain_discriminator_accuracy
                        else:
                            ADV_LOSS = adv_loss(feat[sketch_unreal], feat[sketch_real])
                            domain_acc = adv_loss.domain_discriminator_accuracy
                        writer.add_scalar("train/ADV_LOSS", ADV_LOSS.item(), iter)
                        for i in range(len(domain_acc)):
                            writer.add_scalar(f"train/domain_acc_{i}", domain_acc[i].item(), iter)
                        loss = cfg.MODEL.ADV_LOSS_WEIGHT * ADV_LOSS + loss

                    if 'jmmd' in cfg.MODEL.MODULE:
                        if isinstance(feat, list):
                            JMMD_LOSS=jmmd_loss((feat[1][~sketch],score[0][~sketch]),
                                                (feat[1][sketch],score[0][sketch]))
                            JMMD_LOSS +=jmmd_loss((feat[2][~sketch],score[1][~sketch]),
                                                (feat[2][sketch],score[1][sketch]))
                        else:
                            JMMD_LOSS = jmmd_loss((feat[~sketch],score[~sketch]),
                                                (feat[sketch],score[sketch]))
                        writer.add_scalar("train/JMMD_LOSS", JMMD_LOSS.item(), iter)
                        loss = cfg.MODEL.JMMD_LOSS_WEIGHT * JMMD_LOSS + loss

                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


def make_loss_pf(cfg, num_classes,domainDiscriminator,sigmoid=False):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    
    if 'lwf' in cfg.MODEL.MODULE:
        print("using lwf loss for training")
        old_classes = num_classes
        
        def loss_func_old(score, feat, target, target_cam, i2tscore = None,writer=None,iter=0,sketch_cam=[]):
            loss = F.cross_entropy(score[:,:old_classes], target)
            return loss
        def loss_func_new(score, feat, target, target_cam, i2tscore = None,writer=None,iter=0,sketch_cam=[]):
            lamda = 3
            T = 2
            # fake_targets = target - old_classes
            loss_clf = F.cross_entropy(
                score[0][:, old_classes :], target
            )

            loss_kd = _KD_loss(
                score[0][:, : old_classes],
                score[1][:, : old_classes],
                T,
            )

            loss = lamda * loss_kd + loss_clf

            return loss

        return loss_func_old,loss_func_new,center_criterion

    if 'adv' in cfg.MODEL.MODULE:
        adv_loss=DomainAdversarialLoss(domainDiscriminator,sigmoid=sigmoid)
        print("using adv loss for training")

    if 'jmmd' in cfg.MODEL.MODULE:
        jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=(
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                (GaussianKernel(sigma=0.92, track_running_stats=False),)
            ),
            linear=False, thetas=None
        )
        print("using jmmd loss for training")

    if 'ot' in cfg.MODEL.MODULE:
        print("using ot loss for training")
        ot_loss = OptimalTransport()


    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        # xent_r = CrossEntropyLabelSmooth(num_classes=cfg.DATASETS.NUM_REAL_ID)
        # xent_u = CrossEntropyLabelSmooth(num_classes=num_classes - cfg.DATASETS.NUM_REAL_ID)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore = None,writer=None,iter=0,sketch_cam=[]):
            sketch_cam = torch.tensor(sketch_cam, dtype=target_cam.dtype, device=target_cam.device)
            comparison = target_cam.unsqueeze(-1) == sketch_cam.unsqueeze(0)
            sketch = comparison.any(dim=-1)
            # print('sketch:',sketch)
            # mask_real = target < cfg.DATASETS.NUM_REAL_ID[task]
            # target_r = target[mask_real]
            # target_u = target[~mask_real] - cfg.DATASETS.NUM_REAL_ID[task]
            # sketch_real = sketch & mask_real
            # sketch_unreal = sketch & ~mask_real

            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        for scor in score[0:]:
                            xent(scor, target)
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)
                    if writer is not None:
                        writer.add_scalar("train/ID_LOSS", ID_LOSS.item(), iter)
                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, target)[0]
                    if writer is not None:
                        writer.add_scalar("train/TRI_LOSS", TRI_LOSS.item(), iter)
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    if i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                        if writer is not None:
                            writer.add_scalar("train/I2TLOSS", I2TLOSS.item(), iter)


                    if 'adv' in cfg.MODEL.MODULE:
                        if isinstance(feat, list):
                            ADV_LOSS = [adv_loss(feats[sketch_unreal], feats[sketch_real]) for feats in feat]
                            ADV_LOSS = sum(ADV_LOSS)
                            domain_acc = adv_loss.domain_discriminator_accuracy
                        else:
                            ADV_LOSS = adv_loss(feat[sketch_unreal], feat[sketch_real])
                            domain_acc = adv_loss.domain_discriminator_accuracy
                        if writer is not None:
                            writer.add_scalar("train/ADV_LOSS", ADV_LOSS.item(), iter)
                            for i in range(len(domain_acc)):
                                writer.add_scalar(f"train/domain_acc_{i}", domain_acc[i].item(), iter)
                        loss = cfg.MODEL.ADV_LOSS_WEIGHT * ADV_LOSS + loss

                    if 'jmmd' in cfg.MODEL.MODULE:

                        if isinstance(feat, list):
                            JMMD_LOSS=jmmd_loss((feat[1][~sketch],score[0][~sketch]),
                                                (feat[1][sketch],score[0][sketch]))
                            JMMD_LOSS +=jmmd_loss((feat[2][~sketch],score[1][~sketch]),
                                                (feat[2][sketch],score[1][sketch]))
                        else:
                            JMMD_LOSS = jmmd_loss((feat[~sketch],score[~sketch]),
                                                (feat[sketch],score[sketch]))
                        if writer is not None:
                            writer.add_scalar("train/JMMD_LOSS", JMMD_LOSS.item(), iter)
                        loss = cfg.MODEL.JMMD_LOSS_WEIGHT * JMMD_LOSS + loss

                    if 'ot' in cfg.MODEL.MODULE:
                        if isinstance(feat, list):
                            OT_LOSS = ot_loss(feat[1][~sketch],feat[1][sketch])
                            OT_LOSS += ot_loss(feat[2][~sketch],feat[2][sketch])
                        else:
                            OT_LOSS = ot_loss(feat[~sketch],feat[sketch])
                        if writer is not None:
                            writer.add_scalar("train/OT_LOSS", OT_LOSS.item(), iter)
                        loss = cfg.MODEL.OT_LOSS_WEIGHT * OT_LOSS + loss

                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
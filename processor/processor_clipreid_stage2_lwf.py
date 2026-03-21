import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss


def do_train_stage2_lwf(cfg,
                    model,
                    center_criterion,
                    domain_discriminator,
                    train_loader_stage2,
                    val_loader,
                    optimizer,
                    optimizer_center,
                    scheduler,
                    loss_fn,
                    num_query, local_rank,
                    writer,
                    epoch_start=0,
                    task=999,
                    loss_funces=None,task_name=None,prev_text_features=None,iter_start=0,old_network=None):
    print("start train stage 2")
    # 初始化配置
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    # 设备与分布式设置
    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    if task > 0:
        epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS // 2
        epoch_start = epoch_start - ((cfg.SOLVER.STAGE2.MAX_EPOCHS - epochs)*(task-1))
    
    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if old_network is not None:
            old_network.to(local_rank)
        domain_discriminator.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
        # else:
        #     # num_classes = model.num_classes
        #     num_classes = model.num_classes

    # 初始化统计器和工具
    loss_meter = AverageMeter()  # 损失统计器
    loss_meter_replay = AverageMeter()  # 损失统计器
    acc_meter = AverageMeter()   # 准确率统计器
    acc_meter_replay = AverageMeter()   # 准确率统计器
    evaluator = [R1_mAP_eval(query_num, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM) for query_num in num_query]  # 初始化评估器
    scaler = amp.GradScaler()  # 混合精度梯度缩放器
    xent = SupConLoss(device)  # 监督对比损失实例
    


    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    best_mAP = 0
    best_r1 = 0

    # 训练主循环
    for epoch in range(epoch_start, epoch_start + epochs):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for evaltor in evaluator:
            evaltor.reset()

        scheduler.step()

        model.train()
        domain_discriminator.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            sum_iter=epoch * len(train_loader_stage2) + (n_iter + 1) + iter_start
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view)
                if task==1:
                    old_network.train()
                    with torch.no_grad(): 
                        score_old, feat_old, image_features_old = old_network(x=img, label=target, cam_label=target_cam, view_label=target_view)
                    scores = [score[1],score_old[1]]
                else:
                    scores=score[1]
                loss = loss_fn(scores, None, target, None, None,writer,iter=sum_iter)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (score[1].max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
            writer.add_scalar("train/loss", loss_meter.avg, sum_iter)
            writer.add_scalar("train/acc", acc_meter.avg, sum_iter)
            writer.add_scalar("train/lr", scheduler.get_lr()[0], epoch)
        
        

    # Epoch结束处理
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(os.path.join(cfg.OUTPUT_DIR ,cfg.DATASETS.NAMES,cfg.MODEL.MODULE,cfg.LOG_NAME), cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        

        if epoch % eval_period == 0:
            model.eval()
            for i in range(len(val_loader)):
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader[i]):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else:
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else:
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view,task=i)
                        evaluator[i].update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator[i].compute()
                logger.info("Task {} Validation Results - Epoch: {}".format(i,epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                writer.add_scalar("eval/mAP_task{}".format(i), round(mAP*100,2), epoch)
                writer.add_scalar("eval/r1_task{}".format(i), round(cmc[0]*100,2), epoch)
                writer.add_scalar("eval/r5_task{}".format(i), round(cmc[4]*100,2), epoch)
                writer.add_scalar("eval/r10_task{}".format(i), round(cmc[9]*100,2), epoch)
                torch.cuda.empty_cache()
            
                if mAP > best_mAP:
                    best_mAP = mAP
                    torch.save(model.state_dict(),
                            os.path.join(os.path.join(cfg.OUTPUT_DIR ,cfg.DATASETS.NAMES,cfg.MODEL.MODULE,cfg.LOG_NAME), cfg.MODEL.NAME+'_'+cfg.MODEL.MODULE + 'task_' + str(task) + '_best_mAP.pth'))
                if cmc[0] > best_r1:
                    best_r1 = cmc[0]
                    torch.save(model.state_dict(),
                            os.path.join(os.path.join(cfg.OUTPUT_DIR ,cfg.DATASETS.NAMES,cfg.MODEL.MODULE,cfg.LOG_NAME), cfg.MODEL.NAME+'_'+cfg.MODEL.MODULE + 'task_' + str(task) + '_best_r1.pth'))
            

    # 训练结束处理
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)
    return sum_iter


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
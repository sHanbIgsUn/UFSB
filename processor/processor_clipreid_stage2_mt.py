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


def do_train_stage2_mt(cfg,
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
        domain_discriminator.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes[task]
        else:
            # num_classes = model.num_classes[task]
            num_classes = model.num_classes[task]

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

    # 预计算文本特征
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label=l_list, get_text=True,task=task)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

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
                score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view,task=task)
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits,writer,iter=sum_iter,task=task,task_name=task_name)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()
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
        
        if task > 0 and 'bank' in cfg.MODEL.MODULE:
            batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // 2
            for prev_task in range(task):
                loss_meter_replay.reset()
                acc_meter_replay.reset()

                # 提取历史任务数据
                images_p = torch.stack([image for tuple_list in model.person_bank['photo'].values() 
                                            for image, _, _ in tuple_list])
                images_s = torch.stack([image for tuple_list in model.person_bank['sketch'].values() 
                                            for image, _, _ in tuple_list])
                labels = torch.tensor(list(model.person_bank['sketch'].keys()))
                cams_p = torch.tensor([cam for tuple_list in model.person_bank['photo'].values() 
                                            for _, cam, _ in tuple_list])
                cams_s = torch.tensor([cam for tuple_list in model.person_bank['sketch'].values() 
                                            for _, cam, _ in tuple_list])

                keys_order_p = list(model.person_bank['photo'].keys())
                keys_order_s = list(model.person_bank['sketch'].keys())
                assert keys_order_p == keys_order_s, "image_bank_p 和 image_bank_s 的键顺序不一致"
                
                # 打乱顺序后训练
                indices = torch.randperm(len(labels))
                images_p = images_p[indices]
                images_s = images_s[indices]
                labels = labels[indices]
                cams_p = cams_p[indices]
                cams_s = cams_s[indices]
                for batch_idx in range(0, len(labels), batch_size):
                    re_iter=sum_iter + prev_task * len(labels) + (batch_idx + 1)
                    end_idx = min(batch_idx + batch_size, len(labels))
                    # 提取当前批次
                    batch_p = images_p[batch_idx : end_idx]
                    batch_s = images_s[batch_idx : end_idx]
                    batch_labels = labels[batch_idx : end_idx]
                    batch_cam_p = cams_p[batch_idx : end_idx]
                    batch_cam_s = cams_s[batch_idx : end_idx]

                    if(len(batch_p)<batch_size):
                        continue
                        # num_needed = batch_size - len(batch_p)
                        # indices_nd = torch.randint(0, len(batch_p), (num_needed,))
                        # batch_p = torch.cat([batch_p, images_p[indices_nd]], dim=0)
                        # batch_s = torch.cat([batch_s, images_s[indices_nd]], dim=0)
                        # batch_labels = torch.cat([batch_labels, labels[indices_nd]], dim=0)
                        # batch_cam_p = torch.cat([batch_cam_p, cams_p[indices_nd]], dim=0)
                        # batch_cam_s = torch.cat([batch_cam_s, cams_s[indices_nd]], dim=0)

                    batch_images = torch.cat([batch_p, batch_s], dim=0).to(device)
                    batch_labels = torch.cat([batch_labels, batch_labels],dim=0).to(device)
                    batch_cams = torch.cat([batch_cam_p, batch_cam_s],dim=0).to(device)
                    
                    # 前向传播与训练
                    optimizer.zero_grad()
                    with amp.autocast(enabled=True):
                        score, feat, image_features = model(x=batch_images, label=batch_labels, cam_label=batch_cams, view_label=None,task=prev_task)
                        logits = image_features @ prev_text_features.t()
                        
                        loss = loss_funces[prev_task](score, feat, batch_labels, batch_cams, logits,None,0,prev_task,task_name)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    acc = (logits.max(1)[1] == batch_labels).float().mean()
                    loss_meter_replay.update(loss.item(), batch_images.shape[0])
                    acc_meter_replay.update(acc, 1)

                    torch.cuda.synchronize()

                    writer.add_scalar("train/loss_replay", loss_meter_replay.avg, re_iter)
                    writer.add_scalar("train/acc_replay", acc_meter_replay.avg, re_iter)

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
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
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
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                avg_mAP = 0
                avg_cmc_r1 = 0
                num_tasks = 0
                
                if isinstance(val_loader, list):
                    task_mAPs = []
                    task_r1s = []
                    
                    for i in range(len(val_loader)):
                        evaluator[i].reset() # 重置每个任务的评估器
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
                                feat = model(img, cam_label=camids, view_label=target_view)
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
                        
                        # 收集单个任务的指标
                        task_mAPs.append(mAP)
                        task_r1s.append(cmc[0])
                        avg_mAP += mAP
                        avg_cmc_r1 += cmc[0]
                        num_tasks += 1
                    
                    # 计算平均指标
                    if num_tasks > 0:
                        avg_mAP /= num_tasks
                        avg_cmc_r1 /= num_tasks
                    
                    # 记录平均指标
                    logger.info("Average Validation Results - Epoch: {}".format(epoch))
                    logger.info("Average mAP: {:.1%}".format(avg_mAP))
                    logger.info("Average Rank-1: {:.1%}".format(avg_cmc_r1))
                    
                    # 将平均指标写入tensorboard
                    writer.add_scalar("eval/avg_mAP", round(avg_mAP*100,2), epoch)
                    writer.add_scalar("eval/avg_r1", round(avg_cmc_r1*100,2), epoch)
                    
                else:
                    # 原有的单任务评估逻辑
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
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
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    writer.add_scalar("eval/mAP", round(mAP*100,2), epoch)
                    writer.add_scalar("eval/r1", round(cmc[0]*100,2), epoch)
                    writer.add_scalar("eval/r5", round(cmc[4]*100,2), epoch)
                    writer.add_scalar("eval/r10", round(cmc[9]*100,2), epoch)
                    avg_mAP = mAP
                    avg_cmc_r1 = cmc[0]
                    num_tasks = 1
                    
                torch.cuda.empty_cache()

                # --- 新增的模型保存逻辑 ---
                # 保存基于平均mAP的最佳模型
                if avg_mAP > best_mAP:
                    best_mAP = avg_mAP
                    torch.save(model.state_dict(),
                            os.path.join(os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.NAMES, cfg.MODEL.MODULE, cfg.LOG_NAME), 
                                    cfg.MODEL.NAME + '_' + cfg.MODEL.MODULE + '_best_avg_mAP' + '_' + epoch + '.pth'))
                    logger.info(f"New best average mAP model saved at epoch {epoch}.")

                # 保存基于平均Rank-1的最佳模型
                if avg_cmc_r1 > best_r1:
                    best_r1 = avg_cmc_r1
                    torch.save(model.state_dict(),
                            os.path.join(os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.NAMES, cfg.MODEL.MODULE, cfg.LOG_NAME), 
                                    cfg.MODEL.NAME + '_' + cfg.MODEL.MODULE + '_best_avg_r1' + '_'  + epoch + '.pth'))
                    logger.info(f"New best average Rank-1 model saved at epoch {epoch}.")
            
            

    # 训练结束处理
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)
    return text_features,sum_iter


def do_inference_pf(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")
    
    
    evaluator = [R1_mAP_eval(query_num, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM) for query_num in num_query]
    for evaltor in evaluator:
            evaltor.reset()
    
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
        logger.info("Task {} Validation Results")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    
    return
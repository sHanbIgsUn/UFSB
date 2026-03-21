import logging
import os
import torch
import torch.nn as nn
import time
from datetime import timedelta
from model.clip.prompt_learner_domain import PromptLearner_domain
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank,
             task=0):

    # 日志等参数
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD
    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    # 损失度量
    loss_meter = AverageMeter()  # 初始化损失平均值计量器
    scaler = amp.GradScaler()  # 初始化梯度缩放器，用于混合精度训练
    xent = SupConLoss(device)  # 初始化监督对比损失函数，指定设备
    
    # train

    all_start_time = time.monotonic()
    # logger.info("model: {}".format(model))
    image_features = []
    labels = []
    # 推理获得所有图像特征
    with torch.no_grad():
        # 图像,ID,camID,viewID
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image = True , task=task)
                # 遍历当前批次的每个样本，将特征和标签存入列表
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
                
        labels_list = torch.stack(labels, dim=0).cuda() #N
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features
    print("image_features_list",image_features_list.shape)
    print("train_loader_stage1",64*len(train_loader_stage1))
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            with amp.autocast(enabled=True):
                text_features = model(label = target, get_text = True, task=task)

            # 计算对比损失
            loss_i2t = xent(image_features, text_features, target, target)  # 图像到文本的对比损失
            loss_t2i = xent(text_features, image_features, target, target)  # 文本到图像的对比损失
            loss = loss_i2t + loss_t2i  # 总损失为两者之和

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    # class_ctx_tensor = model.prompt_learner.cls_ctx.data  # 获取参数的数据部分，即普通张量
    # model.classifier_pool.append(PromptLearner_domain(model.cfg_domain, class_ctx_tensor, model.clip_model))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))

from collections import defaultdict
import copy
import random
from model.clip.model import VisionTransformer
import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

from module.ConformalPredictor import ConformalPredictor

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class ImageEncoder(nn.Module):
    def __init__(self,visual):
        super().__init__()
        self.visual=visual
        self.encoder_shared = self.visual.transformer.resblocks[:10]
        self.encoder_real = copy.deepcopy(self.visual.transformer.resblocks[10:])
        self.encoder_unreal = copy.deepcopy(self.visual.transformer.resblocks[10:])
        

    def forward(self, x: torch.Tensor, cv_emb=None, instance_tokens=None,branch='None'):

        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if cv_emb != None:
            x[:, 0] = x[:, 0] + cv_emb

        if instance_tokens is not None:
            instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                   device=x.device)

        x = x + self.visual.positional_embedding.to(x.dtype)

        if instance_tokens is not None:
            x = torch.cat([x[:, :1, :], instance_tokens, x[:, 1:, :]], dim=1)

        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder_shared(x)
        if branch == 'real':
            encoder_domain = self.encoder_real
        elif branch == 'unreal':
            encoder_domain = self.encoder_unreal
        x11 = encoder_domain[0](x)
        x12 = encoder_domain[1](x11)
        x11 = x11.permute(1, 0, 2)  # LND -> NLD  
        x12 = x12.permute(1, 0, 2)  # LND -> NLD  

        x12 = self.visual.ln_post(x12)

        if self.visual.proj is not None:
            xproj = x12 @ self.visual.proj

        return x11, x12, xproj

class build_transformer_pf(nn.Module):
    def __init__(self, num_classes, camera_num, view_num,dateset_order,num_real_id, cfg):
        super(build_transformer_pf, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.num_real_id=num_real_id
        if 'bank' in cfg.MODEL.MODULE:
            self.bank = True
        else:
            self.bank = False
        self.data_bank =  {"sketch":defaultdict(list),"photo":defaultdict(list)}
        self.conformalPredictor=ConformalPredictor()

        self.classifier = nn.ModuleList([nn.Linear(self.in_planes, self.num_classes[0], bias=False),nn.Linear(self.in_planes, self.num_classes[1], bias=False)])
        self.classifier[0].apply(weights_init_classifier)
        self.classifier[1].apply(weights_init_classifier)
        self.classifier_proj = nn.ModuleList([nn.Linear(self.in_planes_proj, self.num_classes[0], bias=False),nn.Linear(self.in_planes_proj, self.num_classes[1], bias=False)])
        self.classifier_proj[0].apply(weights_init_classifier)
        self.classifier_proj[1].apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)


        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = ImageEncoder(clip_model.visual)
        self.count=0
        self.dateset_order=dateset_order
        # if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
        #     self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(camera_num))
        # elif cfg.MODEL.SIE_CAMERA:
        #     self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(camera_num))
        # elif cfg.MODEL.SIE_VIEW:
        #     self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = nn.ModuleList([PromptLearner(num_classes[0], dateset_order[0], clip_model.dtype, clip_model.token_embedding),
                                             PromptLearner(num_classes[1], dateset_order[1], clip_model.dtype, clip_model.token_embedding)])
        self.text_encoder = TextEncoder(clip_model)

        

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None,task=999):
        if get_text == True:
            prompts = self.prompt_learner[task](label)
            text_features = self.text_encoder(prompts, self.prompt_learner[task].tokenized_prompts)
            return text_features

        # 根据 label 划分两个子批次
        mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        if label is not None:
            mask = label < self.num_real_id[task]
        x_real = x[mask]
        x_unreal = x[~mask]  
        batch_size = x.size(0)


        if get_image == True:
            if(x_real is not None and x_real.shape[0] != 0):
                _, _, image_features_proj_r = self.image_encoder(x_real, branch='real')
            if(x_unreal is not None and x_unreal.shape[0] != 0):
                _, _, image_features_proj_u = self.image_encoder(x_unreal, branch='unreal')

            image_features_proj = torch.zeros((batch_size,129, self.in_planes_proj), dtype=image_features_proj_r.dtype, device=image_features_proj_r.device)
            if(x_unreal is not None and x_unreal.shape[0] != 0):
                image_features_proj[~mask] = image_features_proj_u
            image_features_proj[mask] = image_features_proj_r

            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
            
        if self.model_name == 'ViT-B-16':
            # if cam_label != None and view_label!=None:
            #     cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            # elif cam_label != None:
            #     cv_embed = self.sie_coe * self.cv_embed[cam_label]
            # elif view_label!=None:
            #     cv_embed = self.sie_coe * self.cv_embed[view_label]
            # else:
            #     cv_embed = None
            cv_embed_r = None # cv_embed[mask]
            cv_embed_u = None # cv_embed[~mask]

            image_features_last_r, image_features_r, image_features_proj_r = self.image_encoder(x_real,cv_embed_r, branch='real')

            img_feature_last_r = image_features_last_r[:,0]
            img_feature_r = image_features_r[:,0]
            img_feature_proj_r = image_features_proj_r[:,0]


            img_feature_last =  torch.zeros((batch_size, self.in_planes), dtype=img_feature_last_r.dtype, device=img_feature_last_r.device)
            img_feature_last[mask] = img_feature_last_r

            img_feature =  torch.zeros((batch_size, self.in_planes), dtype=img_feature_r.dtype, device=img_feature_r.device)
            img_feature[mask] = img_feature_r

            img_feature_proj = torch.zeros((batch_size,self.in_planes_proj), dtype=img_feature_proj_r.dtype, device=img_feature_proj_r.device)
            img_feature_proj[mask] = img_feature_proj_r

            if x_unreal.shape[0] != 0:
                image_features_last_u, image_features_u, image_features_proj_u = self.image_encoder(x_unreal,cv_embed_u, branch='unreal')

                img_feature_last_u = image_features_last_u[:,0]
                img_feature_u = image_features_u[:,0]
                img_feature_proj_u = image_features_proj_u[:,0]
                

                img_feature_last[~mask] = img_feature_last_u
                img_feature[~mask] = img_feature_u
                img_feature_proj[~mask] = img_feature_proj_u
        
        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 

        if self.training:
            cls_score = self.classifier[task](feat)
            cls_score_proj = self.classifier_proj[task](feat_proj)
            # probabilities = F.softmax(cls_score, dim=1)
            if task==0 and self.bank:
                if self.dateset_order[task]=='person':
                    sketch = [6,7,8,9,10,11,14,15]
                if self.dateset_order[task]=='face':
                    sketch = [0,2]
                batch_size = x.shape[0]
                for i in range(batch_size):
                    current_label = label[i].item()
                    current_cam = cam_label[i].item()
                    
                    if self.dateset_order[task]=='face' and current_label % 8 !=0:
                        continue

                    pred_set,pred_span = self.conformalPredictor.generate_prediction_set(cls_score[i])
                    # if current_cam >= 12: # fake
                    #     continue


                    uncertain_score=len(pred_set)+pred_span
                    # uncertain_score=1

                    if current_cam in sketch:
                        mod = "sketch"
                    else:
                        mod = "photo"
                    if self.data_bank[mod].get(current_label) is None:
                        self.data_bank[mod][current_label] = [(x[i].detach().clone().cpu(),current_cam,uncertain_score)]
                    # 1 last
                    # else: 
                    #     self.data_bank[mod][current_label][0]=(x[i].detach().clone().cpu(),current_cam,uncertain_score)
                    # 1 min
                    elif uncertain_score <= self.data_bank[mod][current_label][0][2]:  
                        self.data_bank[mod][current_label][0]=(x[i].detach().clone().cpu(),current_cam,uncertain_score)
                    
                    # 3
                    # elif len(self.data_bank[mod].get(current_label))<3:
                    #     self.data_bank[mod][current_label].append((x[i].detach().clone().cpu(),current_cam,uncertain_score))
                    # 3 last
                    # else:
                    #     ri = random.choice([0,1,2])
                    #     self.data_bank[mod][current_label][ri]=(x[i].detach().clone().cpu(),current_cam,uncertain_score)
                    # 3 min
                    # else:
                    #     max_i = -1
                    #     max_score = 0
                    #     for idx, (_, _, score) in enumerate(self.data_bank[mod].get(current_label)):
                    #         if score >= max_score:
                    #             max_i = idx
                    #             max_score = score
                    #     if uncertain_score <= max_score:
                            # self.data_bank[mod][current_label][max_i]=(x[i].detach().clone().cpu(),current_cam,uncertain_score)

            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature_r, img_feature_proj_r], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model_pf(cfg, num_classes, view_num, camera,dateset_order,num_real_id):
    model = build_transformer_pf(num_classes, view_num, camera,dateset_order,num_real_id, cfg)
    return model

class build_transformer_ufsb(nn.Module):
    def __init__(self,cfg ,num_classes_list, camera_num_list, view_num_list, dataset_order_list, num_real_id_list, sketch_list):
        """
        Args:
            num_classes_list (list): 每个任务的类别数列表，例如 [num_classes_task1, num_classes_task2]
            camera_num_list (list): 每个任务的相机数列表
            view_num_list (list): 每个任务的视角数列表
            dataset_order_list (list): 每个任务的数据集顺序列表
            num_real_id_list (list): 每个任务的真实ID数列表
        """
        super(build_transformer_ufsb, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.num_tasks = len(num_classes_list) # 动态计算任务数

        # --- 模型尺寸 ---
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        self.num_classes_list = num_classes_list
        self.camera_num_list = camera_num_list
        self.view_num_list = view_num_list
        self.sie_coe = cfg.MODEL.SIE_COE
        self.num_real_id_list = num_real_id_list
        self.sketch_list = sketch_list

        # --- 配置选项 ---
        if 'bank' in cfg.MODEL.MODULE:
            self.bank = True
        else:
            self.bank = False
        self.data_bank = {"sketch": defaultdict(list), "photo": defaultdict(list)}
        self.conformalPredictor = ConformalPredictor()
        self.task_list=cfg.DATASETS.TASK

        # --- 分类器 (Classifier) ---
        # 为每个任务创建一个全连接层
        self.classifier = nn.ModuleList()
        self.classifier_proj = nn.ModuleList()
        for num_cls in num_classes_list:
            linear_cls = nn.Linear(self.in_planes, num_cls, bias=False)
            linear_cls.apply(weights_init_classifier)
            self.classifier.append(linear_cls)

            linear_cls_proj = nn.Linear(self.in_planes_proj, num_cls, bias=False)
            linear_cls_proj.apply(weights_init_classifier)
            self.classifier_proj.append(linear_cls_proj)

        # --- 瓶颈层 (Neck / BN) ---
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # --- CLIP 模型加载 ---
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = ImageEncoder(clip_model.visual)

        self.dataset_order_list = dataset_order_list

        # --- 提示学习器 (Prompt Learner) ---
        # 为每个任务创建一个 PromptLearner
        self.prompt_learner = nn.ModuleList()
        for i in range(self.num_tasks):
            prompt_learner = PromptLearner(
                num_classes_list[i],
                dataset_order_list[i],
                clip_model.dtype,
                clip_model.token_embedding
            )
            self.prompt_learner.append(prompt_learner)

        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None,task=999):
        if get_text == True:
            prompts = self.prompt_learner[task](label)
            text_features = self.text_encoder(prompts, self.prompt_learner[task].tokenized_prompts)
            return text_features

        # 根据 label 划分两个子批次
        mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        if label is not None:
            mask = label < self.num_real_id_list[task]
        x_real = x[mask]
        x_unreal = x[~mask]  
        batch_size = x.size(0)


        if get_image == True:
            if(x_real is not None and x_real.shape[0] != 0):
                _, _, image_features_proj_r = self.image_encoder(x_real, branch='real')
                image_features_proj = torch.zeros((batch_size,129, self.in_planes_proj), dtype=image_features_proj_r.dtype, device=image_features_proj_r.device)
            if(x_unreal is not None and x_unreal.shape[0] != 0):
                _, _, image_features_proj_u = self.image_encoder(x_unreal, branch='unreal')
                image_features_proj = torch.zeros((batch_size,129, self.in_planes_proj), dtype=image_features_proj_u.dtype, device=image_features_proj_u.device)
            
            if(x_real is not None and x_real.shape[0] != 0):
                image_features_proj[mask] = image_features_proj_r

            if(x_unreal is not None and x_unreal.shape[0] != 0):
                image_features_proj[~mask] = image_features_proj_u


            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
            
        if self.model_name == 'ViT-B-16':
            # if cam_label != None and view_label!=None:
            #     cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            # elif cam_label != None:
            #     cv_embed = self.sie_coe * self.cv_embed[cam_label]
            # elif view_label!=None:
            #     cv_embed = self.sie_coe * self.cv_embed[view_label]
            # else:
            #     cv_embed = None
            cv_embed_r = None # cv_embed[mask]
            cv_embed_u = None # cv_embed[~mask]

            image_features_last_r, image_features_r, image_features_proj_r = self.image_encoder(x_real,cv_embed_r, branch='real')

            img_feature_last_r = image_features_last_r[:,0]
            img_feature_r = image_features_r[:,0]
            img_feature_proj_r = image_features_proj_r[:,0]


            img_feature_last =  torch.zeros((batch_size, self.in_planes), dtype=img_feature_last_r.dtype, device=img_feature_last_r.device)
            img_feature_last[mask] = img_feature_last_r

            img_feature =  torch.zeros((batch_size, self.in_planes), dtype=img_feature_r.dtype, device=img_feature_r.device)
            img_feature[mask] = img_feature_r

            img_feature_proj = torch.zeros((batch_size,self.in_planes_proj), dtype=img_feature_proj_r.dtype, device=img_feature_proj_r.device)
            img_feature_proj[mask] = img_feature_proj_r

            if x_unreal.shape[0] != 0:
                image_features_last_u, image_features_u, image_features_proj_u = self.image_encoder(x_unreal,cv_embed_u, branch='unreal')

                img_feature_last_u = image_features_last_u[:,0]
                img_feature_u = image_features_u[:,0]
                img_feature_proj_u = image_features_proj_u[:,0]
                

                img_feature_last[~mask] = img_feature_last_u
                img_feature[~mask] = img_feature_u
                img_feature_proj[~mask] = img_feature_proj_u
        
        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 

        if self.training:
            cls_score = self.classifier[task](feat)
            cls_score_proj = self.classifier_proj[task](feat_proj)
            # probabilities = F.softmax(cls_score, dim=1)
            if task==0 and self.bank:
                sketch = self.sketch_list[task]
                batch_size = x.shape[0]
                for i in range(batch_size):
                    current_label = label[i].item()
                    current_cam = cam_label[i].item()
                    
                    # if self.dateset_order[task]=='face' and current_label % 8 !=0: face数据太多只选择1/8的id
                    #     continue
                    if current_label >= self.num_real_id_list[task]: # fake
                        continue
                    pred_set,pred_span = self.conformalPredictor.generate_prediction_set(cls_score[i])
                    uncertain_score=len(pred_set)+pred_span
                    # uncertain_score=1

                    if current_cam in sketch:
                        mod = "sketch"
                    else:
                        mod = "photo"
                    if self.data_bank[mod].get(current_label) is None:
                        self.data_bank[mod][current_label] = [(x[i].detach().clone().cpu(),current_cam,uncertain_score)]
                    # 1 last
                    # else: 
                    #     self.data_bank[mod][current_label][0]=(x[i].detach().clone().cpu(),current_cam,uncertain_score)
                    # 1 min
                    elif uncertain_score <= self.data_bank[mod][current_label][0][2]:  
                        self.data_bank[mod][current_label][0]=(x[i].detach().clone().cpu(),current_cam,uncertain_score)
                    
                    # 3
                    # elif len(self.data_bank[mod].get(current_label))<3:
                    #     self.data_bank[mod][current_label].append((x[i].detach().clone().cpu(),current_cam,uncertain_score))
                    # 3 last
                    # else:
                    #     ri = random.choice([0,1,2])
                    #     self.data_bank[mod][current_label][ri]=(x[i].detach().clone().cpu(),current_cam,uncertain_score)
                    # 3 min
                    # else:
                    #     max_i = -1
                    #     max_score = 0
                    #     for idx, (_, _, score) in enumerate(self.data_bank[mod].get(current_label)):
                    #         if score >= max_score:
                    #             max_i = idx
                    #             max_score = score
                    #     if uncertain_score <= max_score:
                            # self.data_bank[mod][current_label][max_i]=(x[i].detach().clone().cpu(),current_cam,uncertain_score)

            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature_r, img_feature_proj_r], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model




class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "ms1k":
            ctx_init = "A photo of a X X X X person."
        else:
            ctx_init = "A photo of a X X X X face."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 


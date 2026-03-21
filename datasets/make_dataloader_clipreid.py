from datasets.cufsf import CUFSF
from datasets.cufsf_celeba_s import CUFSF_CelebA_S
from datasets.fake_reid import Fake_reid
from datasets.forensic_sketch import Forensic_sketch
from datasets.iiit_d_celeba_s import IIIT_D_CelebA_S
from datasets.iiit_d_sketch import IIIT_D_sketch
from datasets.ms1k_cufsf import MS1k_CUFSF
from datasets.ms1k_cufsf_fake import MS1k_CUFSF_Fake
from datasets.ms1k_fake import MS1k_Fake, MS1k_Fake_mt
from datasets.ms1k_iiit_d_fake import MS1k_IIIT_Fake
from datasets.wild_sketch import WildSketch
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomFaceSampler_SI, RandomFaceSampler_SIRU, RandomIdentitySampler, RandomIdentitySampler_SI, RandomIdentitySampler_SIRU, RandomPFSampler_SIRU

from .market1501 import Market1501
from .ms1k import MS1k

from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist


__factory = {
    'market1501': Market1501,
    'ms1k': MS1k,
    'fake_reid': Fake_reid,
    'ms1k_fake': MS1k_Fake,
    'ms1k_cufsf': MS1k_CUFSF,
    'ms1k_cufsf_fake': MS1k_CUFSF_Fake,
    'ms1k_iiit_fake': MS1k_IIIT_Fake,
    'ms1k_fake_mt': MS1k_Fake_mt,
    'cufsf': CUFSF,
    'cufsf_fake': CUFSF_CelebA_S,
    'wild': WildSketch,
    'forensic': Forensic_sketch,
    'iiit': IIIT_D_sketch,
    'iiit_fake': IIIT_D_CelebA_S,
}

__dataset_reid = ['ms1k']
__dataset_face = ['cufsf','iiit','wild','forensic']


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_real_pid = dataset.num_real_pid
    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            if cfg.DATASETS.NAMES == 'ms1k_fake':
                print('Dataloadar SIRU')
                sampler = RandomIdentitySampler_SIRU(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                               cfg.DATALOADER.NUM_INSTANCE)
            elif cfg.DATASETS.NAMES == 'ms1k_cufsf_fake' or cfg.DATASETS.NAMES == 'ms1k_iiit_fake':
                print('sampler RandomPFSampler_SIRU')
                sampler = RandomPFSampler_SIRU(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                cfg.DATALOADER.NUM_INSTANCE,dataset.num_real_pid,dataset.sketch_cam)
            elif 'ms1k' in cfg.DATASETS.NAMES or 'fake_reid' in cfg.DATASETS.NAMES:
                print('Dataloadar SI')
                sampler = RandomIdentitySampler_SI(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                               cfg.DATALOADER.NUM_INSTANCE)
            elif 'face' in cfg.DATASETS.NAMES or 'iiit' in cfg.DATASETS.NAMES:
                if 'fake' in cfg.DATASETS.NAMES:
                    print('Dataloadar SIRU')
                    sampler = RandomFaceSampler_SIRU(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                               cfg.DATALOADER.NUM_INSTANCE,cfg.DATASETS.NUM_REAL_ID[0])
                else:
                    print('Dataloadar SI')
                    sampler = RandomFaceSampler_SI(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                               cfg.DATALOADER.NUM_INSTANCE)
            else:
                print('Dataloadar other')
                sampler = RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                                   cfg.DATALOADER.NUM_INSTANCE)
                
            train_loader_stage2 = DataLoader(
                train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                sampler=sampler,
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    if cfg.DATASETS.NAMES == 'ms1k_cufsf_fake' or cfg.DATASETS.NAMES == 'ms1k_iiit_fake' or cfg.DATASETS.NAMES == 'ms1k_cufsf':
        val_set_person = ImageDataset(dataset.query_person + dataset.gallery_person, val_transforms)
        val_set_face = ImageDataset(dataset.query_face + dataset.gallery_face, val_transforms)
        val_loader = [] 
        val_loader.append(DataLoader(
            val_set_person, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        ))
        val_loader.append(DataLoader(
            val_set_face, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        ))
        num_query=[len(dataset.query_person),len(dataset.query_face)]
    else:
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        val_loader = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        num_query=len(dataset.query)

    
    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )
    return train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, cam_num, view_num,num_real_pid


def make_dataloader_mt(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    assert 'mt' in cfg.DATASETS.NAMES, 'mt should be in the dataset name'

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR,num_tasks=cfg.DATASETS.NUM_TASKS)
    ############## for multi-task dataloader ##############
    train_set = [ImageDataset(train, train_transforms) for train in dataset.train]
    train_set_normal = [ImageDataset(train, val_transforms) for train in dataset.train]
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            if 'ms1k' or 'fake_reid' in cfg.DATASETS.NAMES:
                sampler = [RandomIdentitySampler_SI(train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                               cfg.DATALOADER.NUM_INSTANCE) for train in dataset.train]
            else:
                sampler = RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                                   cfg.DATALOADER.NUM_INSTANCE)
            train_loader_stage2 = [DataLoader(
                train_set[i], batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                sampler=sampler[i],
                num_workers=num_workers, collate_fn=train_collate_fn
            ) for i in range(cfg.DATASETS.NUM_TASKS)]
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_stage1 = [DataLoader(
        train_set_normal[i], batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    ) for i in range(cfg.DATASETS.NUM_TASKS)]
    return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num

def make_dataloader_pf(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    if 'ms1k' in cfg.DATASETS.NAMES:
        if 'fake' in cfg.DATASETS.NAMES:
            dataset_person = __factory['ms1k_fake'](root=cfg.DATASETS.ROOT_DIR)
        else:
            dataset_person = __factory['ms1k'](root=cfg.DATASETS.ROOT_DIR)
    
    if 'cufsf' in cfg.DATASETS.NAMES:
        if 'fake' in cfg.DATASETS.NAMES:
            dataset_face = __factory['cufsf_fake'](root=cfg.DATASETS.ROOT_DIR)
        else:
            dataset_face = __factory['cufsf_face'](root=cfg.DATASETS.ROOT_DIR)
    elif 'iiit_d' in cfg.DATASETS.NAMES:
        if 'fake' in cfg.DATASETS.NAMES:
            dataset_face = __factory['iiit_fake'](root=cfg.DATASETS.ROOT_DIR)
        else:
            dataset_face = __factory['iiit_face'](root=cfg.DATASETS.ROOT_DIR)

    real_id_person = dataset_person.num_real_pid
    real_id_face = dataset_face.num_real_pid

    print("DATASETS.NUM_REAL_ID:",real_id_person,real_id_face)

    train_set_person = ImageDataset(dataset_person.train, train_transforms) 
    train_set_normal_person = ImageDataset(dataset_person.train, val_transforms)
    train_set_face = ImageDataset(dataset_face.train, train_transforms)
    train_set_normal_face = ImageDataset(dataset_face.train, val_transforms)
    
    num_classes_person = dataset_person.num_train_pids
    cam_num_person = dataset_person.num_train_cams
    view_num_person = dataset_person.num_train_vids
    num_classes_face = dataset_face.num_train_pids
    cam_num_face = dataset_face.num_train_cams
    view_num_face = dataset_face.num_train_vids

    if 'fake' in cfg.DATASETS.NAMES:
        sampler_person = RandomIdentitySampler_SIRU(dataset_person.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                               cfg.DATALOADER.NUM_INSTANCE,max_real_id=real_id_person)
        sampler_face = RandomFaceSampler_SIRU(dataset_face.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                               cfg.DATALOADER.NUM_INSTANCE,max_real_id=real_id_face)
    else:
        sampler_person = RandomIdentitySampler_SI(dataset_person.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                                    cfg.DATALOADER.NUM_INSTANCE)
        sampler_face = RandomFaceSampler_SI(dataset_face.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                              cfg.DATALOADER.NUM_INSTANCE)
    train_loader_stage2_person = DataLoader(
        train_set_person, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
        sampler=sampler_person,
        num_workers=num_workers, collate_fn=train_collate_fn
    )
    train_loader_stage2_face = DataLoader(
        train_set_face, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
        sampler=sampler_face,
        num_workers=num_workers, collate_fn=train_collate_fn
    )

    val_set_person = ImageDataset(dataset_person.query + dataset_person.gallery, val_transforms)
    val_set_face = ImageDataset(dataset_face.query + dataset_face.gallery, val_transforms)

    val_loader_person = DataLoader(
        val_set_person, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    val_loader_face = DataLoader(
        val_set_face, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_stage1_person = DataLoader(
        train_set_normal_person, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )
    train_loader_stage1_face = DataLoader(
        train_set_normal_face, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )

    return train_loader_stage2_person, train_loader_stage2_face, train_loader_stage1_person, train_loader_stage1_face, val_loader_person, val_loader_face, len(dataset_person.query), len(dataset_face.query), num_classes_person, num_classes_face, cam_num_person, cam_num_face, view_num_person,view_num_face,real_id_person,real_id_face

def make_dataloader_ufsb(cfg):
    """
    动态创建多任务数据加载器。
    根据 cfg.DATASETS.TASK 动态加载不同任务的数据集。
    """
    # 解析任务列表
    task_names = cfg.DATASETS.TASK
    print(f"Configured tasks: {task_names}")
    
    # 初始化存储容器
    all_datasets = []
    all_train_loaders_stage2 = []
    all_train_loaders_stage1 = []
    all_val_loaders = []
    all_num_queries = []
    all_num_classes_list = []
    all_cam_nums = []
    all_view_nums = []
    all_real_ids = []
    all_sketch = []

    # 为每个任务创建加载器和元数据
    for i, task_name in enumerate(task_names):
        print(f"Loading data for task: {task_name}")
        
        # --- 数据变换 ---
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

        num_workers = cfg.DATALOADER.NUM_WORKERS

        # 处理 fake 数据集
        final_dataset_key = f"{task_name}_fake" if 'fake' in cfg.DATASETS.NAMES else task_name
        
        # --- 加载数据集 ---
        try:
            dataset = __factory[final_dataset_key](root=cfg.DATASETS.ROOT_DIR)
        except KeyError:
            raise KeyError(f"Dataset factory does not contain key '{final_dataset_key}'. Check your dataset configuration and factory registration.")
        
        all_datasets.append(dataset)

        # --- 提取元数据 ---
        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids
        real_id = getattr(dataset, 'num_real_pid', 0) # 假设数据集有此属性，否则默认为0
        if task_name in __dataset_reid:
            sketch = [6,7,8,9,10,11,14,15]
        elif task_name in __dataset_face:
            sketch = [0,2]

        all_num_classes_list.append(num_classes)
        all_cam_nums.append(cam_num)
        all_view_nums.append(view_num)
        all_real_ids.append(real_id)
        all_sketch.append(sketch)

        # --- 创建数据集子集 ---
        train_set_stage2 = ImageDataset(dataset.train, train_transforms)
        train_set_stage1 = ImageDataset(dataset.train, val_transforms) # For stage1, normal shuffle
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        all_num_queries.append(len(dataset.query)) # Assuming num_query is length of query set

        # --- 创建采样器 (根据是否有'fake'和任务类型) ---
        if 'fake' in cfg.DATASETS.NAMES:
            if task_name in __dataset_reid:
                sampler = RandomIdentitySampler_SIRU(
                    dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE, max_real_id=real_id
                )
            elif task_name in __dataset_face: # Add other face-like tasks here
                # Note: This assumes 'max_real_id' logic is handled correctly per task.
                # It might need adjustment based on how datasets are structured.
                sampler = RandomFaceSampler_SIRU(
                    dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE, max_real_id=real_id
                )
            else:
                # Default fallback for unknown task types with fake data
                sampler = RandomIdentitySampler_SIRU(
                    dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
                )
        else:
            # No fake data, use SI samplers
            if task_name in ['ms1k']:
                sampler = RandomIdentitySampler_SI(
                    dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
                )
            elif task_name in ['cufsf','iiit','wild','forensic']:
                sampler = RandomFaceSampler_SI(
                    dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
                )
            else:
                # Default fallback for unknown task types without fake data
                sampler = RandomIdentitySampler_SI(
                    dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
                )

        # --- 创建 DataLoader ---
        train_loader_stage2 = DataLoader(
            train_set_stage2, 
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            sampler=sampler,
            num_workers=num_workers, 
            collate_fn=train_collate_fn
        )
        all_train_loaders_stage2.append(train_loader_stage2)

        train_loader_stage1 = DataLoader(
            train_set_stage1, 
            batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, 
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=train_collate_fn
        )
        all_train_loaders_stage1.append(train_loader_stage1)

        val_loader = DataLoader(
            val_set, 
            batch_size=cfg.TEST.IMS_PER_BATCH, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        all_val_loaders.append(val_loader)

        print(f"  Task '{task_name}' loaded. Classes: {num_classes}, Cams: {cam_num}, Views: {view_num}, Real IDs: {real_id}")

    # 5. 打包返回值
    all_metadata = (
        all_num_queries,
        all_num_classes_list,
        all_cam_nums,
        all_view_nums,
        all_real_ids,
        all_sketch
    )
    
    print(f"DATASETS.NUM_REAL_ID for all tasks:", all_real_ids)

    return (
        all_train_loaders_stage2, 
        all_train_loaders_stage1,
        all_val_loaders, 
        *all_metadata
    )
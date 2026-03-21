import glob
import os
import re

import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.datautils.core50data import CORE50
import os.path as osp

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iGanFake(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args["class_order"]
        self.class_order = class_order

    def download_data(self):

        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'train')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'val')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)


class iCore50(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(8 * 50).tolist()
        self.class_order = class_order

    def download_data(self):
        datagen = CORE50(root=self.args["data_path"], scenario="ni")

        dataset_list = []
        for i, train_batch in enumerate(datagen):
            imglist, labellist = train_batch
            labellist += i * 50
            imglist = imglist.astype(np.uint8)
            dataset_list.append([imglist, labellist])
        train_x = np.concatenate(np.array(dataset_list)[:, 0])
        train_y = np.concatenate(np.array(dataset_list)[:, 1])
        self.train_data = train_x
        self.train_targets = train_y

        test_x, test_y = datagen.get_test_set()
        test_x = test_x.astype(np.uint8)
        self.test_data = test_x
        self.test_targets = test_y


class iDomainNet(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(6 * 345).tolist()
        self.class_order = class_order
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]

    def download_data(self):
        self.image_list_root = self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.test_data = np.array(train_x)
        self.test_targets = np.array(train_y)


class iMarket_sketch1k(iData):
    use_path = True
    # 数据预处理配置
    train_trsf = [
        transforms.RandomResizedCrop((256,128)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize((256,128)),
        # transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    # 初始化
    def __init__(self, args):
        self.args = args
        class_order = np.arange(498 + 1467).tolist()
        self.class_order = class_order
        self.domain_names = ["real", "unreal"]


    def download_data(self):
        # 路径配置
        self.dataset_dir =self.args["data_path"]
        self.train_dir_photo = osp.join(self.dataset_dir, 'real', 'photo', 'train')
        self.train_dir_sketch = osp.join(self.dataset_dir, 'real', 'sketch', 'train')
        self.query_dir = osp.join(self.dataset_dir, 'real', 'sketch', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'real', 'photo', 'query')


        # 测试集加载（查询集 + 画廊集）
        img_paths = []
        for root, dirs, files in os.walk(self.query_dir):
            for file in files:
                if file.endswith('.jpg'):
                    img_paths.append(osp.join(root, file))
        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(osp.basename(img_path)[:-4])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        test_x = []
        test_y = []
        for img_path in sorted(img_paths):
            pid = int(osp.basename(img_path)[:-4])
            camid = int(osp.basename(osp.dirname(img_path)))
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 7 <= camid <= 12
            camid -= 1  # index starts from 0

            test_x.append(img_path)
            test_y.append(pid)


        img_paths = glob.glob(osp.join(self.gallery_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        # 加载画廊集（照片）
        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            test_x.append(img_path)
            test_y.append(pid)

        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)

        # 训练集加载（照片 + 素描）
        photo_img_paths = glob.glob(osp.join(self.train_dir_photo, '*.jpg'))
        sketch_img_paths = []
        for root, dirs, files in os.walk(self.train_dir_sketch):
            for file in files:
                if file.endswith('.jpg'):
                    sketch_img_paths.append(osp.join(root, file))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for sketch_img_path in sorted(sketch_img_paths):
            pid = int(osp.basename(sketch_img_path)[:-4])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        train_x = []
        train_y = []
        for photo_img_path in sorted(photo_img_paths):
            pid, camid = map(int, pattern.search(photo_img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0

            train_x.append(photo_img_path)
            train_y.append(pid)
        for sketch_img_path in sorted(sketch_img_paths):
            pid = int(osp.basename(sketch_img_path)[:-4])
            camid = int(osp.basename(osp.dirname(sketch_img_path)))
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 7 <= camid <= 12
            camid -= 1  # index starts from 0

            train_x.append(sketch_img_path)
            train_y.append(pid)

        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)


import glob
import os
import os.path as osp
import re

from .bases import BaseImageDataset


class IIIT_D_CelebA_S(BaseImageDataset):
    """

    """
    dataset_dir_r = 'IIIT-D_sketch'
    dataset_dir_f = 'CelebA-Sketch'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(IIIT_D_CelebA_S, self).__init__()
        self.dataset_dir_r = osp.join(root, self.dataset_dir_r)
        self.dataset_dir_f = osp.join(root, self.dataset_dir_f)
        self.train_dir_photo_r = osp.join(self.dataset_dir_r, 'Photo', 'train')
        self.train_dir_sketch_r = osp.join(self.dataset_dir_r, 'Sketch', 'train')
        self.train_dir_photo_f = osp.join(self.dataset_dir_f, 'CelebA_cropped_mtcnn_all')
        self.train_dir_sketch_f = osp.join(self.dataset_dir_f, 'CelebA_cropped_sketch_all')
        self.query_dir = osp.join(self.dataset_dir_r, 'Sketch', 'query')
        self.gallery_dir = osp.join(self.dataset_dir_r, 'Photo', 'query')
        self.num_real_pid=0

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir_train(self.train_dir_photo_r, self.train_dir_sketch_r, self.train_dir_photo_f, self.train_dir_sketch_f, relabel=True)
        query = self._process_dir_query(self.query_dir, relabel=True)
        gallery = self._process_dir_gallery(self.gallery_dir, relabel=True)

        if verbose:
            print("=> IIIT-D_CelebA_S loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir_r):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir_r))
        if not osp.exists(self.dataset_dir_f):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir_f))
        if not osp.exists(self.train_dir_photo_r):
            raise RuntimeError("'{}' is not available".format(self.train_dir_photo_r))
        if not osp.exists(self.train_dir_photo_f):
            raise RuntimeError("'{}' is not available".format(self.train_dir_photo_f))
        if not osp.exists(self.train_dir_sketch_r):
            raise RuntimeError("'{}' is not available".format(self.train_dir_sketch_r))
        if not osp.exists(self.train_dir_sketch_f):
            raise RuntimeError("'{}' is not available".format(self.train_dir_sketch_f))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_query(self, dir_path, relabel=False):
        img_paths = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.jpg'):
                    img_paths.append(osp.join(root, file))
        pid_container = set()
        for img_path in sorted(img_paths):
            pid = osp.basename(img_path)[:-4]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid = osp.basename(img_path)[:-4]
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            camid = 0
            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset

    def _process_dir_gallery(self, dir_path, relabel=False):
        img_paths = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.jpg'):
                    img_paths.append(osp.join(root, file))
        pid_container = set()
        for img_path in sorted(img_paths):
            pid = osp.basename(img_path)[:-4]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid = osp.basename(img_path)[:-4]
            if relabel: pid = pid2label[pid]
            camid = 1
            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset

    def _process_dir_train(self, train_dir_photo_r, train_dir_sketch_r, train_dir_photo_f, train_dir_sketch_f, relabel=False):
        # real img path 
        sketch_img_paths_r = []
        for root, dirs, files in os.walk(train_dir_sketch_r):
            for file in files:
                if file.endswith('.jpg'):
                    sketch_img_paths_r.append(osp.join(root, file))

        photo_img_paths_r = []
        for root, dirs, files in os.walk(train_dir_photo_r):
            for file in files:
                if file.endswith('.jpg'):
                    photo_img_paths_r.append(osp.join(root, file))

        # fake img path
        sketch_img_paths_f = []
        photo_img_paths_f = []
        for root, dirs, files in os.walk(train_dir_sketch_f):
            for file in files:
                if file.endswith('.jpg'):
                    sketch_img_paths_f.append(osp.join(root, file))
        for root, dirs, files in os.walk(train_dir_photo_f):
            for file in files:
                if file.endswith('.jpg'):
                    photo_img_paths_f.append(osp.join(root, file))

        # print("photo_img_paths_r: ", photo_img_paths_r[0])
        # print("sketch_img_paths_r: ", sketch_img_paths_r[0])
        # print("photo_img_paths_f: ", photo_img_paths_f[0])
        # print("sketch_img_paths_f: ", sketch_img_paths_f[0])
        
        
        pid_container = set()
        # real pid
        real_pid_container = set()
        for sketch_img_path in sorted(sketch_img_paths_r):
            pid = osp.basename(sketch_img_path)[:-4]
            real_pid_container.add(pid)
        
        real_pid2label = {pid: label for label, pid in enumerate(real_pid_container)}
        for sketch_img_path in sorted(sketch_img_paths_r):
            pid = osp.basename(sketch_img_path)[:-4]
            pid_container.add(real_pid2label[pid])
        
        max_real_pid = max(pid_container)
        self.num_real_pid=len(pid_container)
        # fake pid
        name_to_id = {}
        with open(osp.join(self.dataset_dir_f,'identity_CelebA.txt'), 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()  # 去除首尾空白
                if not line:
                    continue  # 跳过空行
                
                # 拆分图片名和ID（支持任意空白符分隔）
                img_name, img_id_str = line.split()
                img_id = int(img_id_str)  # 尝试将ID转换为整数
                name_to_id[img_name] = img_id + max_real_pid  # 存储映射关系

        for sketch_img_path in sorted(sketch_img_paths_f):
            pid = name_to_id[osp.basename(sketch_img_path)]
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for sketch_img_path in sorted(sketch_img_paths_r):
            pid = osp.basename(sketch_img_path)[:-4]
            camid = 0
            if pid == -1: continue  # junk images are just ignored
            if real_pid2label[pid] not in pid_container: continue
            if relabel: pid = pid2label[real_pid2label[pid]]
            dataset.append((sketch_img_path, self.pid_begin + pid, camid, 0))
        for photo_img_path in sorted(photo_img_paths_r):
            pid = osp.basename(photo_img_path)[:-4]
            camid = 1
            if real_pid2label[pid] not in pid_container: continue
            if relabel: pid = pid2label[real_pid2label[pid]]
            dataset.append((photo_img_path, self.pid_begin + pid, camid, 0))

        for sketch_img_path in sorted(sketch_img_paths_f):
            pid = name_to_id[osp.basename(sketch_img_path)]
            camid = 2
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            if relabel: pid = pid2label[pid]
            dataset.append((sketch_img_path, self.pid_begin + pid, camid, 0))
        for photo_img_path in sorted(photo_img_paths_f):
            pid = name_to_id[osp.basename(photo_img_path)]
            camid = 3
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            if relabel: pid = pid2label[pid]
            dataset.append((photo_img_path, self.pid_begin + pid, camid, 0))
            
        return dataset
import glob
import os
import os.path as osp
import re

from .bases import BaseImageDataset


class IIIT_D_sketch(BaseImageDataset):
    """

    """
    dataset_dir_r = 'IIIT-D_sketch'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(IIIT_D_sketch, self).__init__()
        self.dataset_dir_r = osp.join(root, self.dataset_dir_r)
        self.train_dir_photo_r = osp.join(self.dataset_dir_r, 'Photo', 'train')
        self.train_dir_sketch_r = osp.join(self.dataset_dir_r, 'Sketch', 'train')
        self.query_dir = osp.join(self.dataset_dir_r, 'Sketch', 'query')
        self.gallery_dir = osp.join(self.dataset_dir_r, 'Photo', 'query')
        self.num_real_pid = 0
        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir_train(self.train_dir_photo_r, self.train_dir_sketch_r, relabel=True)
        query = self._process_dir_query(self.query_dir, relabel=True)
        gallery = self._process_dir_gallery(self.gallery_dir, relabel=True)

        if verbose:
            print("=> IIIT_D_sketch loaded")
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
        if not osp.exists(self.train_dir_photo_r):
            raise RuntimeError("'{}' is not available".format(self.train_dir_photo_r))
        if not osp.exists(self.train_dir_sketch_r):
            raise RuntimeError("'{}' is not available".format(self.train_dir_sketch_r))
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

    def _process_dir_train(self, train_dir_photo_r, train_dir_sketch_r, relabel=False):
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

        # real pid
        pid_container = set()
        for sketch_img_path in sorted(sketch_img_paths_r):
            pid = osp.basename(sketch_img_path)[:-4]
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        self.num_real_pid = len(pid_container)
        dataset = []
        for sketch_img_path in sorted(sketch_img_paths_r):
            pid = osp.basename(sketch_img_path)[:-4]
            camid = 0
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            if relabel: pid = pid2label[pid]
            dataset.append((sketch_img_path, self.pid_begin + pid, camid, 0))
        for photo_img_path in sorted(photo_img_paths_r):
            pid = osp.basename(photo_img_path)[:-4]
            camid = 1
            if pid not in pid_container: continue
            if relabel: pid = pid2label[pid]
            dataset.append((photo_img_path, self.pid_begin + pid, camid, 0))
            
        return dataset
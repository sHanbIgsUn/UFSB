import glob
import os
import os.path as osp
import re

from .bases import BaseImageDataset


class Fake_reid(BaseImageDataset):
    """

    """
    dataset_dir = 'Fake_reid'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Fake_reid, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir_photo = osp.join(self.dataset_dir, 'photo', 'train')
        self.train_dir_sketch = osp.join(self.dataset_dir, 'sketch', 'train')
        self.query_dir = osp.join(self.dataset_dir, 'sketch', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'photo', 'query')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir_train(self.train_dir_photo, self.train_dir_sketch, relabel=True)
        query = self._process_dir_query(self.query_dir, relabel=False)
        gallery = self._process_dir_gallery(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Fake_reid loaded")
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
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir_photo):
            raise RuntimeError("'{}' is not available".format(self.train_dir_photo))
        if not osp.exists(self.train_dir_sketch):
            raise RuntimeError("'{}' is not available".format(self.train_dir_sketch))
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
            pid = int(osp.basename(osp.dirname(img_path)))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid = int(osp.basename(osp.dirname(img_path)))
            camid = int(osp.basename(osp.dirname(osp.dirname(img_path))))
            if pid == -1: continue  # junk images are just ignored
            assert 1502 <= pid <= 2234  # pid == 0 means background
            assert 15 <= camid <= 16
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

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
            pid = int(osp.basename(osp.dirname(img_path)))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid = int(osp.basename(osp.dirname(img_path)))
            camid = int(osp.basename(osp.dirname(osp.dirname(img_path))))
            if pid == -1: continue  # junk images are just ignored
            assert 1502 <= pid <= 2234  # pid == 0 means background
            assert 13 <= camid <= 14
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset

    def _process_dir_train(self, train_dir_photo, train_dir_sketch, relabel=False):
        photo_img_paths = []
        sketch_img_paths = []
        for root, dirs, files in os.walk(train_dir_photo):
            for file in files:
                if file.endswith('.jpg'):
                    photo_img_paths.append(osp.join(root, file))
        for root, dirs, files in os.walk(train_dir_sketch):
            for file in files:
                if file.endswith('.jpg'):
                    sketch_img_paths.append(osp.join(root, file))

        pid_container = set()
        for sketch_img_path in sorted(sketch_img_paths):
            pid = int(osp.basename(osp.dirname(sketch_img_path)))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for photo_img_path in sorted(photo_img_paths):
            pid = int(osp.basename(osp.dirname(photo_img_path)))
            camid = int(osp.basename(osp.dirname(osp.dirname(photo_img_path))))
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert 2235 <= pid <= 2968  # pid == 0 means background
            assert 13 <= camid <= 14
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((photo_img_path, self.pid_begin + pid, camid, 0))
        for sketch_img_path in sorted(sketch_img_paths):
            pid = int(osp.basename(osp.dirname(sketch_img_path)))
            camid = int(osp.basename(osp.dirname(osp.dirname(sketch_img_path))))
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert 2235 <= pid <= 2968  # pid == 0 means background
            assert 15 <= camid <= 16
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((sketch_img_path, self.pid_begin + pid, camid, 0))
        return dataset

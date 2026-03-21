import glob
import os
import os.path as osp
import re

from .bases import BaseImageDataset


class MS1k_Fake(BaseImageDataset):
    """

    """
    dataset_dir_r = 'MS1k'
    dataset_dir_f = 'Fake_reid'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MS1k_Fake, self).__init__()
        self.dataset_dir_r = osp.join(root, self.dataset_dir_r)
        self.dataset_dir_f = osp.join(root, self.dataset_dir_f)
        self.train_dir_photo_r = osp.join(self.dataset_dir_r, 'photo', 'train')
        self.train_dir_sketch_r = osp.join(self.dataset_dir_r, 'sketch', 'train')
        self.train_dir_photo_f = osp.join(self.dataset_dir_f, 'photo', 'train')
        self.train_dir_sketch_f = osp.join(self.dataset_dir_f, 'sketch', 'train')
        self.query_dir = osp.join(self.dataset_dir_r, 'sketch', 'query')
        self.gallery_dir = osp.join(self.dataset_dir_r, 'photo', 'query')
        self.num_real_pid=0

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir_train(self.train_dir_photo_r, self.train_dir_sketch_r, self.train_dir_photo_f, self.train_dir_sketch_f, relabel=True)
        query = self._process_dir_query(self.query_dir, relabel=False)
        gallery = self._process_dir_gallery(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MS1k_Fake loaded")
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
            pid = int(osp.basename(img_path)[:-4])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid = int(osp.basename(img_path)[:-4])
            camid = int(osp.basename(osp.dirname(img_path)))
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 7 <= camid <= 12
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset

    def _process_dir_gallery(self, dir_path, relabel=False):
        # person
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 0))

        return dataset

    def _process_dir_train(self, train_dir_photo_r, train_dir_sketch_r, train_dir_photo_f, train_dir_sketch_f, relabel=False):
        # real dataset
        photo_img_paths_r = glob.glob(osp.join(train_dir_photo_r, '*.jpg'))
        sketch_img_paths_r = []
        for root, dirs, files in os.walk(train_dir_sketch_r):
            for file in files:
                if file.endswith('.jpg'):
                    sketch_img_paths_r.append(osp.join(root, file))

        # fake dataset
        photo_img_paths_f = []
        sketch_img_paths_f = []
        for root, dirs, files in os.walk(train_dir_photo_f):
            for file in files:
                if file.endswith('.jpg'):
                    photo_img_paths_f.append(osp.join(root, file))
        for root, dirs, files in os.walk(train_dir_sketch_f):
            for file in files:
                if file.endswith('.jpg'):
                    sketch_img_paths_f.append(osp.join(root, file))
        
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for sketch_img_path in sorted(sketch_img_paths_r):
            pid = int(osp.basename(sketch_img_path)[:-4])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        self.num_real_pid=len(pid_container)
        for sketch_img_path in sorted(sketch_img_paths_f):
            pid = int(osp.basename(osp.dirname(sketch_img_path)))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for photo_img_path in sorted(photo_img_paths_r):
            pid, camid = map(int, pattern.search(photo_img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((photo_img_path, self.pid_begin + pid, camid, 0))
        for sketch_img_path in sorted(sketch_img_paths_r):
            pid = int(osp.basename(sketch_img_path)[:-4])
            camid = int(osp.basename(osp.dirname(sketch_img_path)))
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 7 <= camid <= 12
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((sketch_img_path, self.pid_begin + pid, camid, 0))

        for photo_img_path in sorted(photo_img_paths_f):
            pid = int(osp.basename(osp.dirname(photo_img_path)))
            camid = int(osp.basename(osp.dirname(osp.dirname(photo_img_path))))
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert 2235 <= pid <= 2968  # pid == 0 means background
            assert 13 <= camid <= 14
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((photo_img_path, self.pid_begin + pid, camid, 0))
        for sketch_img_path in sorted(sketch_img_paths_f):
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


class MS1k_Fake_mt(BaseImageDataset):
    """

    """
    dataset_dir_r = 'MS1k'
    dataset_dir_f = 'Fake_reid'

    def __init__(self, root='', verbose=True, pid_begin=0,num_tasks=5, **kwargs):
        super(MS1k_Fake_mt, self).__init__()
        self.dataset_dir_r = osp.join(root, self.dataset_dir_r)
        self.dataset_dir_f = osp.join(root, self.dataset_dir_f)
        self.train_dir_photo_r = osp.join(self.dataset_dir_r, 'photo', 'train')
        self.train_dir_sketch_r = osp.join(self.dataset_dir_r, 'sketch', 'train')
        self.train_dir_photo_f = osp.join(self.dataset_dir_f, 'photo', 'train')
        self.train_dir_sketch_f = osp.join(self.dataset_dir_f, 'sketch', 'train')
        self.query_dir = osp.join(self.dataset_dir_r, 'sketch', 'query')
        self.gallery_dir = osp.join(self.dataset_dir_r, 'photo', 'query')

        self.num_tasks = num_tasks

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir_train(self.train_dir_photo_r, self.train_dir_sketch_r, self.train_dir_photo_f, self.train_dir_sketch_f, relabel=True)
        query = self._process_dir_query(self.query_dir, relabel=False)
        gallery = self._process_dir_gallery(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids=[]
        self.num_train_imgs=[]
        self.num_train_cams=[]
        self.num_train_vids=[]
        
        for tra in self.train:
            num_train_pids,num_train_imgs,num_train_cams,num_train_vids = self.get_imagedata_info(tra)
            self.num_train_pids.append(num_train_pids)
            self.num_train_imgs.append(num_train_imgs)
            self.num_train_cams.append(num_train_cams)
            self.num_train_vids.append(num_train_vids)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)
        
        if verbose:
            print("=> MS1k_Fake_mt loaded")
            self.print_dataset_statistics_mt()
        num_train_cams =0 
        for num_train_cam in self.num_train_cams:
            num_train_cams+=num_train_cam
        self.num_train_cams= num_train_cams

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
        
    def print_dataset_statistics_mt(self):
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        for i in range(self.num_tasks):
            print("  train task"+str(i)+"| {:5d} | {:8d} | {:9d}".format(self.num_train_pids[i], self.num_train_imgs[i], self.num_train_cams[i]))
        print("  query    | {:5d} | {:8d} | {:9d}".format(self.num_query_pids, self.num_query_imgs, self.num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams))
        print("  ----------------------------------------")

    def _process_dir_query(self, dir_path, relabel=False):
        img_paths = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.jpg'):
                    img_paths.append(osp.join(root, file))
        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(osp.basename(img_path)[:-4])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid = int(osp.basename(img_path)[:-4])
            camid = int(osp.basename(osp.dirname(img_path)))
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 7 <= camid <= 12
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset

    def _process_dir_gallery(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset

    def _process_dir_train(self, train_dir_photo_r, train_dir_sketch_r, train_dir_photo_f, train_dir_sketch_f, relabel=False):
        # real dataset
        photo_img_paths_r = glob.glob(osp.join(train_dir_photo_r, '*.jpg'))
        sketch_img_paths_r = []
        for root, dirs, files in os.walk(train_dir_sketch_r):
            for file in files:
                if file.endswith('.jpg'):
                    sketch_img_paths_r.append(osp.join(root, file))

        # fake dataset
        photo_img_paths_f = []
        sketch_img_paths_f = []
        for root, dirs, files in os.walk(train_dir_photo_f):
            for file in files:
                if file.endswith('.jpg'):
                    photo_img_paths_f.append(osp.join(root, file))
        for root, dirs, files in os.walk(train_dir_sketch_f):
            for file in files:
                if file.endswith('.jpg'):
                    sketch_img_paths_f.append(osp.join(root, file))
        
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container_r = set()
        pid_container_f = set()
        pid_containers = []

        for sketch_img_path in sorted(sketch_img_paths_r):
            pid = int(osp.basename(sketch_img_path)[:-4])
            if pid == -1: continue  # junk images are just ignored
            pid_container_r.add(pid)
        for sketch_img_path in sorted(sketch_img_paths_f):
            pid = int(osp.basename(osp.dirname(sketch_img_path)))
            if pid == -1: continue  # junk images are just ignored
            pid_container_f.add(pid)

        pid2label = []
        
        pid_container_f = list(pid_container_f)
        pid_container_r = list(pid_container_r)
        
        nf=len(pid_container_f)//self.num_tasks
        nr=len(pid_container_r)//self.num_tasks
        for i in range(self.num_tasks-1):
            pid_container=pid_container_f[i*nf:(i+1)*nf]
            pid_container.extend(pid_container_r[i*nr:(i+1)*nr])
            pid_containers.append(pid_container)
            print(pid_container)
        pid_containers.append(pid_container_f[(self.num_tasks-1)*nf:])
        pid_containers[-1].extend(pid_container_r[(self.num_tasks-1)*nr:])
        for pids in pid_containers:
            # print("task"+str(len(pids)))
            # print(pids)
            pid2label.append({pid: label for label, pid in enumerate(pids)})

        datasets = []
        
        for i in range(self.num_tasks):
            dataset = []
            for photo_img_path in sorted(photo_img_paths_r):
                pid, camid = map(int, pattern.search(photo_img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                if pid not in pid_containers[i]: continue
                assert 0 <= pid <= 1501  # pid == 0 means background
                assert 1 <= camid <= 6
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[i][pid]
                dataset.append((photo_img_path, self.pid_begin + pid, camid, 0))
            for sketch_img_path in sorted(sketch_img_paths_r):
                pid = int(osp.basename(sketch_img_path)[:-4])
                camid = int(osp.basename(osp.dirname(sketch_img_path)))
                if pid == -1: continue  # junk images are just ignored
                if pid not in pid_containers[i]: continue
                assert 0 <= pid <= 1501  # pid == 0 means background
                assert 7 <= camid <= 12
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[i][pid]
                dataset.append((sketch_img_path, self.pid_begin + pid, camid, 0))

            for photo_img_path in sorted(photo_img_paths_f):
                pid = int(osp.basename(osp.dirname(photo_img_path)))
                camid = int(osp.basename(osp.dirname(osp.dirname(photo_img_path))))
                if pid == -1: continue  # junk images are just ignored
                if pid not in pid_containers[i]: continue
                assert 2235 <= pid <= 2968  # pid == 0 means background
                assert 13 <= camid <= 14
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[i][pid]
                dataset.append((photo_img_path, self.pid_begin + pid, camid, 0))
            for sketch_img_path in sorted(sketch_img_paths_f):
                pid = int(osp.basename(osp.dirname(sketch_img_path)))
                camid = int(osp.basename(osp.dirname(osp.dirname(sketch_img_path))))
                if pid == -1: continue  # junk images are just ignored
                if pid not in pid_containers[i]: continue
                assert 2235 <= pid <= 2968  # pid == 0 means background
                assert 15 <= camid <= 16
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[i][pid]
                dataset.append((sketch_img_path, self.pid_begin + pid, camid, 0))
            datasets.append(dataset)
        return datasets
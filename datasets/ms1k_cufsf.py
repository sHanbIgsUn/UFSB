import glob
import os
import os.path as osp
import re

from .bases import BaseImageDataset


class MS1k_CUFSF(BaseImageDataset):
    """

    """
    person_dir_r = 'MS1k'
    person_dir_f = 'Fake_reid'

    face_dir_r = 'CUFSF'
    face_dir_f = 'CelebA-Sketch'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MS1k_CUFSF, self).__init__()
        # person
        self.person_dir_r = osp.join(root, self.person_dir_r)
        self.train_person_photo_r = osp.join(self.person_dir_r, 'photo', 'train')
        self.train_person_sketch_r = osp.join(self.person_dir_r, 'sketch', 'train')
        self.query_person = osp.join(self.person_dir_r, 'sketch', 'query')
        self.gallery_person = osp.join(self.person_dir_r, 'photo', 'query')
        # face
        self.face_dir_r = osp.join(root, self.face_dir_r)
        self.train_face_photo_r = osp.join(self.face_dir_r, 'photo', 'train')
        self.train_face_sketch_r = osp.join(self.face_dir_r, 'sketch', 'train')
        self.query_face = osp.join(self.face_dir_r, 'sketch', 'query')
        self.gallery_face = osp.join(self.face_dir_r, 'photo', 'query')

        self.sketch_cam=[6,7,8,9,10,11,14,15,16,18]
        self.num_real_pid=0
        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir_train(self.train_person_photo_r, self.train_person_sketch_r,self.train_face_photo_r, self.train_face_sketch_r,  relabel=True)
        query_person = self._process_dir_query_person(self.query_person, relabel=False)
        gallery_person = self._process_dir_gallery_person(self.gallery_person, relabel=False)
        query_face = self._process_dir_query_face(self.query_face, relabel=False)
        gallery_face = self._process_dir_gallery_face(self.gallery_face, relabel=False)

        if verbose:
            print("=> MS1k_CUFSF loaded")
            self.print_dataset_statistics(train, query_person, gallery_person)
            self.print_dataset_statistics(train, query_face, gallery_face)
            print('sketch_cam',self.sketch_cam)

        self.train = train
        self.query_person = query_person
        self.gallery_person = gallery_person
        self.query_face = query_face
        self.gallery_face = gallery_face

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids_person, self.num_query_imgs_person, self.num_query_cams_person, self.num_query_vids_person = self.get_imagedata_info(
            self.query_person)
        self.num_gallery_pids_face, self.num_gallery_imgs_face, self.num_gallery_cams_face, self.num_gallery_vids_face = self.get_imagedata_info(
            self.gallery_person)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        # if not osp.exists(self.dataset_dir_r):
        #     raise RuntimeError("'{}' is not available".format(self.dataset_dir_r))
        # if not osp.exists(self.dataset_dir_f):
        #     raise RuntimeError("'{}' is not available".format(self.dataset_dir_f))
        # if not osp.exists(self.train_dir_photo_r):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir_photo_r))
        # if not osp.exists(self.train_dir_photo_f):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir_photo_f))
        # if not osp.exists(self.train_dir_sketch_r):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir_sketch_r))
        # if not osp.exists(self.train_dir_sketch_f):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir_sketch_f))
        # if not osp.exists(self.query_dir):
        #     raise RuntimeError("'{}' is not available".format(self.query_dir))
        # if not osp.exists(self.gallery_dir):
        #     raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_query_person(self, dir_path, relabel=False):
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

    def _process_dir_gallery_person(self, dir_path, relabel=False):
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
    
    def _process_dir_query_face(self, dir_path, relabel=False):
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
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1209  # pid == 0 means background
            if relabel: pid = pid2label[pid]
            camid = 0
            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset

    def _process_dir_gallery_face(self, dir_path, relabel=False):
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
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1209  # pid == 0 means background
            if relabel: pid = pid2label[pid]
            camid = 1
            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset

    def _process_dir_train(self, train_dir_photo_r_ps, train_dir_sketch_r_ps, train_dir_photo_r_fc, train_dir_sketch_r_fc,  relabel=False):
        # person
        # real dataset
        photo_img_paths_r_ps = glob.glob(osp.join(train_dir_photo_r_ps, '*.jpg'))
        sketch_img_paths_r_ps = []
        for root, dirs, files in os.walk(train_dir_sketch_r_ps):
            for file in files:
                if file.endswith('.jpg'):
                    sketch_img_paths_r_ps.append(osp.join(root, file))

        # face
        # real img path 
        sketch_img_paths_r_fc = []
        for root, dirs, files in os.walk(train_dir_sketch_r_fc):
            for file in files:
                if file.endswith('.jpg'):
                    sketch_img_paths_r_fc.append(osp.join(root, file))

        photo_img_paths_r_fc = []
        for root, dirs, files in os.walk(train_dir_photo_r_fc):
            for file in files:
                if file.endswith('.jpg'):
                    photo_img_paths_r_fc.append(osp.join(root, file))

        
        # person
        # real pid
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for sketch_img_path in sorted(sketch_img_paths_r_ps):
            pid = int(osp.basename(sketch_img_path)[:-4])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        max_pid_person_real = max(pid_container)
        # face
        # real pid
        for sketch_img_path in sorted(sketch_img_paths_r_fc):
            pid = int(osp.basename(sketch_img_path)[:-4]) + max_pid_person_real
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        
        max_pid_real = max(pid_container)
        self.num_real_pid=max_pid_real

        max_pid = max(pid_container)
        
        
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        print(max_pid_person_real,max_pid_real,max_pid)
        dataset = []
        # person
        for photo_img_path in sorted(photo_img_paths_r_ps):
            pid, camid = map(int, pattern.search(photo_img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((photo_img_path, self.pid_begin + pid, camid, 0))
        for sketch_img_path in sorted(sketch_img_paths_r_ps):
            pid = int(osp.basename(sketch_img_path)[:-4])
            camid = int(osp.basename(osp.dirname(sketch_img_path)))
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 7 <= camid <= 12
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((sketch_img_path, self.pid_begin + pid, camid, 0))

        # face
        for sketch_img_path in sorted(sketch_img_paths_r_fc):
            pid = int(osp.basename(sketch_img_path)[:-4]) + max_pid_person_real
            camid = 13
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert max_pid_person_real <= pid <= 1209 + max_pid_person_real  # pid == 0 means background
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((sketch_img_path, self.pid_begin + pid, camid, 0))
        for photo_img_path in sorted(photo_img_paths_r_fc):
            pid = int(osp.basename(photo_img_path)[:-4]) + max_pid_person_real
            camid = 14
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_container: continue
            assert max_pid_person_real <= pid <= 1209 + max_pid_person_real  # pid == 0 means background
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((photo_img_path, self.pid_begin + pid, camid, 0))

        return dataset

from time import sleep
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

class RandomIdentitySampler_SI(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_S = defaultdict(list)
        self.index_dic_I = defaultdict(list)

        for index, (_, pid, camid, _) in enumerate(self.data_source):
            if camid in [0,1,2,3,4,5,12,13]:
                self.index_dic_I[pid].append(index)
            else:
                self.index_dic_S[pid].append(index)
        self.pids = list(self.index_dic_S.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic_S[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_S = copy.deepcopy(self.index_dic_S[pid])
            if len(idxs_I) < self.num_instances // 2 and len(idxs_S) < self.num_instances // 2:
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_S = np.random.choice(idxs_S, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_S):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_S), replace=False)
            if len(idxs_S) > len(idxs_I):
                idxs_S = np.random.choice(idxs_S, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_S)
            batch_idxs = []
            for idx_I, idx_S in zip(idxs_I, idxs_S):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_S)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
                    continue
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
    
class RandomIdentitySampler_SIRU(Sampler):
    def __init__(self, data_source, batch_size, num_instances, max_real_id):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_S = defaultdict(list)
        self.index_dic_I = defaultdict(list)
        self.real_pids = set()
        self.unreal_pids = set()

        for index, (_, pid, camid, _) in enumerate(self.data_source):
            if camid in [0,1,2,3,4,5,12,13]:
                self.index_dic_I[pid].append(index)
            else:
                self.index_dic_S[pid].append(index)
            if pid < max_real_id:
                self.real_pids.add(pid)
            else:
                self.unreal_pids.add(pid)
        
        self.real_pids = list(self.real_pids)
        self.unreal_pids = list(self.unreal_pids)
        self.pids = self.real_pids + self.unreal_pids

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs_S = self.index_dic_S[pid]
            idxs_I = self.index_dic_I[pid]
            num = min(len(idxs_S), len(idxs_I))
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_S = copy.deepcopy(self.index_dic_S[pid])
            if len(idxs_I) < self.num_instances // 2 and len(idxs_S) < self.num_instances // 2:
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_S = np.random.choice(idxs_S, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_S):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_S), replace=False)
            if len(idxs_S) > len(idxs_I):
                idxs_S = np.random.choice(idxs_S, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_S)
            batch_idxs = []
            for idx_I, idx_S in zip(idxs_I, idxs_S):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_S)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        
        avai_real_pids = copy.deepcopy(self.real_pids)
        avai_unreal_pids = copy.deepcopy(self.unreal_pids)
        random.shuffle(avai_real_pids)
        random.shuffle(avai_unreal_pids)

        # avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_real_pids) >= self.num_pids_per_batch//2 and len(avai_unreal_pids) >= self.num_pids_per_batch//2:
            for real_pid, unreal_pid in zip(avai_real_pids, avai_unreal_pids):
                real_batch_idxs = batch_idxs_dict[real_pid].pop(0)
                final_idxs.extend(real_batch_idxs)
                if len(batch_idxs_dict[real_pid]) == 0:
                    avai_real_pids.remove(real_pid)
                
                unreal_batch_idxs = batch_idxs_dict[unreal_pid].pop(0)
                final_idxs.extend(unreal_batch_idxs)
                if len(batch_idxs_dict[unreal_pid]) == 0:
                    avai_unreal_pids.remove(unreal_pid)

                if len(avai_real_pids) < self.num_pids_per_batch//2 or len(avai_unreal_pids) < self.num_pids_per_batch//2:
                    break
        
        self.length = len(final_idxs)
        print("image_num of epoch",self.length)
        return iter(final_idxs)

    def __len__(self):
        return self.length
    
class RandomFaceSampler_SI(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_S = defaultdict(list)
        self.index_dic_I = defaultdict(list)

        for index, (_, pid, camid, _) in enumerate(self.data_source):
            if camid in [1,3]:
                self.index_dic_I[pid].append(index)
            else:
                self.index_dic_S[pid].append(index)
        self.pids = list(self.index_dic_S.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic_S[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_S = copy.deepcopy(self.index_dic_S[pid])
            if len(idxs_I) < self.num_instances // 2 and len(idxs_S) < self.num_instances // 2:
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_S = np.random.choice(idxs_S, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_S):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_S), replace=False)
            if len(idxs_S) > len(idxs_I):
                idxs_S = np.random.choice(idxs_S, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_S)
            batch_idxs = []
            for idx_I, idx_S in zip(idxs_I, idxs_S):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_S)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                # if len(batch_idxs_dict[pid]) == 0:
                #     avai_pids.remove(pid)
                #     continue
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
    
class RandomFaceSampler_SIRU(Sampler):
    def __init__(self, data_source, batch_size, num_instances,max_real_id):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_S = defaultdict(list)
        self.index_dic_I = defaultdict(list)
        self.real_pids = set()
        self.unreal_pids = set()

        for index, (_, pid, camid, _) in enumerate(self.data_source):
            if camid in [1,3]:
                self.index_dic_I[pid].append(index)
            else:
                self.index_dic_S[pid].append(index)
            if pid < max_real_id:
                self.real_pids.add(pid)
            else:
                self.unreal_pids.add(pid)
        
        self.real_pids = list(self.real_pids)
        self.unreal_pids = list(self.unreal_pids)
        self.pids = self.real_pids + self.unreal_pids

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs_S = self.index_dic_S[pid]
            idxs_I = self.index_dic_I[pid]
            num = min(len(idxs_S), len(idxs_I))
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_S = copy.deepcopy(self.index_dic_S[pid])
            if len(idxs_I) < self.num_instances // 2 and len(idxs_S) < self.num_instances // 2:
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_S = np.random.choice(idxs_S, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_S):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_S), replace=False)
            if len(idxs_S) > len(idxs_I):
                idxs_S = np.random.choice(idxs_S, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_S)
            batch_idxs = []
            for idx_I, idx_S in zip(idxs_I, idxs_S):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_S)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        
        avai_real_pids = copy.deepcopy(self.real_pids)
        avai_unreal_pids = copy.deepcopy(self.unreal_pids)
        random.shuffle(avai_real_pids)
        random.shuffle(avai_unreal_pids)

        # avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_real_pids) >= self.num_pids_per_batch//2 and len(avai_unreal_pids) >= self.num_pids_per_batch//2:
            for real_pid, unreal_pid in zip(avai_real_pids, avai_unreal_pids):
                real_batch_idxs = batch_idxs_dict[real_pid].pop(0)
                final_idxs.extend(real_batch_idxs)
                if len(batch_idxs_dict[real_pid]) == 0:
                    avai_real_pids.remove(real_pid)
                
                unreal_batch_idxs = batch_idxs_dict[unreal_pid].pop(0)
                final_idxs.extend(unreal_batch_idxs)
                if len(batch_idxs_dict[unreal_pid]) == 0:
                    avai_unreal_pids.remove(unreal_pid)

                if len(avai_real_pids) < self.num_pids_per_batch//2 or len(avai_unreal_pids) < self.num_pids_per_batch//2:
                    break
        
        self.length = len(final_idxs)
        print("image_num of epoch",self.length)
        return iter(final_idxs)

    def __len__(self):
        return self.length
    
class RandomPFSampler_SIRU(Sampler):
    def __init__(self, data_source, batch_size, num_instances,max_real_id,sketch_cam):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_S = defaultdict(list)
        self.index_dic_I = defaultdict(list)
        self.real_pids = set()
        self.unreal_pids = set()

        for index, (_, pid, camid, _) in enumerate(self.data_source):
            if camid not in sketch_cam:
                self.index_dic_I[pid].append(index)
            else:
                self.index_dic_S[pid].append(index)
            if pid < max_real_id:
                self.real_pids.add(pid)
            else:
                self.unreal_pids.add(pid)
        
        self.real_pids = list(self.real_pids)
        self.unreal_pids = list(self.unreal_pids)
        self.pids = self.real_pids + self.unreal_pids

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs_S = self.index_dic_S[pid]
            idxs_I = self.index_dic_I[pid]
            num = min(len(idxs_S), len(idxs_I))
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_S = copy.deepcopy(self.index_dic_S[pid])
            if len(idxs_I) < self.num_instances // 2 and len(idxs_S) < self.num_instances // 2:
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_S = np.random.choice(idxs_S, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_S):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_S), replace=False)
            if len(idxs_S) > len(idxs_I):
                idxs_S = np.random.choice(idxs_S, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_S)
            batch_idxs = []
            for idx_I, idx_S in zip(idxs_I, idxs_S):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_S)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        
        avai_real_pids = copy.deepcopy(self.real_pids)
        avai_unreal_pids = copy.deepcopy(self.unreal_pids)
        random.shuffle(avai_real_pids)
        random.shuffle(avai_unreal_pids)

        # avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_real_pids) >= self.num_pids_per_batch//2 and len(avai_unreal_pids) >= self.num_pids_per_batch//2:
            for real_pid, unreal_pid in zip(avai_real_pids, avai_unreal_pids):
                if len(batch_idxs_dict[real_pid]) == 0:
                    avai_real_pids.remove(real_pid)
                    # print(real_pid)
                    continue
                real_batch_idxs = batch_idxs_dict[real_pid].pop(0)
                final_idxs.extend(real_batch_idxs)
                if len(batch_idxs_dict[real_pid]) == 0:
                    avai_real_pids.remove(real_pid)
                
                unreal_batch_idxs = batch_idxs_dict[unreal_pid].pop(0)
                final_idxs.extend(unreal_batch_idxs)
                if len(batch_idxs_dict[unreal_pid]) == 0:
                    avai_unreal_pids.remove(unreal_pid)

                if len(avai_real_pids) < self.num_pids_per_batch//2 or len(avai_unreal_pids) < self.num_pids_per_batch//2:
                    break
        
        self.length = len(final_idxs)
        print("image_num of epoch",self.length)
        return iter(final_idxs)
    def __len__(self):
        return self.length
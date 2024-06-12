import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torch_geometric as pyg
import numpy as np
from pathlib import Path
import random
import re
import pygmtools as pygm
import time
import copy

# cls_list = ['bur', 'chr', 'els', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai', 'tho', 'wil']
cls_list = ['erdos']
class BaseDataset:
    def __init__(self):
        pass

    def get_pair(self, cls, shuffle):
        raise NotImplementedError

class QAPLIB(BaseDataset):
    def __init__(self, sets, cls, fetch_online=False):
        super(QAPLIB, self).__init__()
        self.classes = ['qaplib']
        self.sets = sets

        if cls is not None and cls != 'none':
            idx = cls_list.index(cls)
            self.cls_list = [cls_list[idx]]
        else:
            self.cls_list = cls_list

        self.data_list = []
        self.qap_path = Path('./synthetic_data/erdos20_0.6/')
        # self.qap_path = Path('./data/qapdata')
        for inst in self.cls_list:
            for dat_path in self.qap_path.glob(inst + '*.dat'):
                name = dat_path.name[:-4]
                prob_size = int(re.findall(r"\d+", name)[0])
                if (self.sets == 'test' and prob_size > 90) \
                    or (self.sets == 'train' and prob_size > 1000):
                    continue
                self.data_list.append(name)

        # remove trivial instance esc16f
        if 'esc16f' in self.data_list:
            self.data_list.remove('esc16f')

        # define compare function
        def name_cmp(a, b):
            a = re.findall(r'[0-9]+|[a-z]+', a)
            b = re.findall(r'[0-9]+|[a-z]+', b)
            for _a, _b in zip(a, b):
                if _a.isdigit() and _b.isdigit():
                    _a = int(_a)
                    _b = int(_b)
                cmp = (_a > _b) - (_a < _b)
                if cmp != 0:
                    return cmp
            if len(a) > len(b):
                return -1
            elif len(a) < len(b):
                return 1
            else:
                return 0

        def cmp_to_key(mycmp):
            'Convert a cmp= function into a key= function'
            class K:
                def __init__(self, obj, *args):
                    self.obj = obj
                def __lt__(self, other):
                    return mycmp(self.obj, other.obj) < 0
                def __gt__(self, other):
                    return mycmp(self.obj, other.obj) > 0
                def __eq__(self, other):
                    return mycmp(self.obj, other.obj) == 0
                def __le__(self, other):
                    return mycmp(self.obj, other.obj) <= 0
                def __ge__(self, other):
                    return mycmp(self.obj, other.obj) >= 0
                def __ne__(self, other):
                    return mycmp(self.obj, other.obj) != 0
            return K

        # sort data list according to the names
        self.data_list.sort(key=cmp_to_key(name_cmp))

    def __len__(self):
        return len(self.data_list)
    
    def get_pair(self, idx, shuffle=None):
        """
        Get QAP data by index
        :param idx: dataset index
        :param shuffle: no use here
        :return: (pair of data, groundtruth permutation matrix)
        """
        name = self.data_list[idx]

        dat_path = self.qap_path / (name + '.dat')
        if Path.exists(self.qap_path / (name + '.sln')):
            sln_path = self.qap_path / (name + '_random_init_random_LNS.sln')
            sln_file = sln_path.open()
        else:
            sln_file = None
        dat_file = dat_path.open()
        

        def split_line(x):
            for _ in re.split(r'[,\s]', x.rstrip('\n')):
                if _ == "":
                    continue
                else:
                    yield float(_)

        dat_list = [[_ for _ in split_line(line)] for line in dat_file]
        if sln_file != None:
            sln_list = [[_ for _ in line] for line in sln_file]
        else:
            sln_list = None

        prob_size = dat_list[0][0]

        # read data
        r = 0
        r_position = 0
        c = 0
        Fi = [[]]
        Fj = [[]]
        postions = [[]]
        F = Fi
        for l in dat_list[1:]:
            F[r] += l
            c += len(l)
            # assert c <= prob_size
            if c == prob_size:
                r += 1
                r_position += 1
                if r < prob_size:
                    F.append([])
                    c = 0
                else:
                    F = Fj
                    r = 0
                    c = 0
                if r_position == 2*prob_size:
                    F = postions
                    r = 0
                    c = 0
        Fi = np.array(Fi, dtype=np.float32)
        Fj = np.array(Fj, dtype=np.float32)
        locations = np.zeros((int(prob_size),2),dtype=np.float32)
        for i in range(len(postions[0])//2):
            locations[i][0],locations[i][1] = postions[0][2*i],postions[0][2*i+1]
        
        # import pdb;pdb.set_trace()
        assert Fi.shape == Fj.shape == (prob_size, prob_size)
        #K = np.kron(Fj, Fi)

        # read solution
        if sln_list != None:
            # import pdb; pdb.set_trace()
            sol = sln_list[0][1]
            obj = sln_list[0][-1]
            perm_list = []
            for _ in sln_list[1:]:
                perm_list += _
            assert len(perm_list) == prob_size
            perm_mat = np.zeros((prob_size, prob_size), dtype=np.float32)
            for r, c in enumerate(perm_list):
                perm_mat[r, c - 1] = 1

            return Fi, Fj, perm_mat, sol, name,obj,locations
        else:
            return Fi,Fj,None, None ,name, None,locations
        
class QAPDataset(Dataset):
    def __init__(self, name, length, cls=None, **args):
        self.name = name
        self.ds = eval(self.name)(**args, cls=cls)
        self.classes = self.ds.classes
        self.cls = None if cls == 'none' else cls
        self.length = length

    def __len__(self):
        #return len(self.ds.data_list)
        return self.length

    def __getitem__(self, idx):
        Fi, Fj, perm_mat, sol, name, obj = self.ds.get_pair(idx % len(self.ds.data_list))

        #if np.max(ori_aff_mat) > 0:
        #    norm_aff_mat = ori_aff_mat / np.mean(ori_aff_mat)
        #else:
        #    norm_aff_mat = ori_aff_mat

        ret_dict = {'Fi': Fi,
                    'Fj': Fj,
                    'name': name}

        return ret_dict


if __name__ == '__main__':
    train_set = QAPLIB('train','erdos')
    F,D,per,sol,name, opt_obj,locations = train_set.get_pair(1)
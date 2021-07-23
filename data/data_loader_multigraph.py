import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from data.pascal_voc import PascalVOC
from data.willow_obj import WillowObject
from data.SPair71k import SPair71k
from utils.build_graphs import build_graphs
import cmath
from utils.config import cfg
from torch_geometric.data import Data, Batch

from scipy import spatial
from scipy.spatial import Delaunay
import math

datasets = {"PascalVOC": PascalVOC,
            "WillowObject": WillowObject,
            "SPair71k": SPair71k}

class GMDataset(Dataset):
    def __init__(self, name, length, **args):
        self.name = name
        self.ds = datasets[name](**args)
        self.true_epochs = length is None
        self.length = (
            self.ds.total_size if self.true_epochs else length
        )  # NOTE images pairs are sampled randomly, so there is no exact definition of dataset size
        if self.true_epochs:
            print(f"Initializing {self.ds.sets}-set with all {self.length} examples.")
        else:
            print(f"Initializing {self.ds.sets}-set. Randomly sampling {self.length} examples.")
        # length here represents the iterations between two checkpoints
        # if length is None the length is set to the size of the ds
        self.obj_size = self.ds.obj_resize
        self.classes = self.ds.classes
        self.cls = None
        self.num_graphs_in_matching_instance = None

    def set_cls(self, cls):
        if cls == "none":
            cls = None
        self.cls = cls
        if self.true_epochs:  # Update length of dataset for dataloader according to class
            self.length = self.ds.total_size if cls is None else self.ds.size_by_cls[cls]

    def _cal_box(self, points):
        x_min = float("inf")
        x_max = -float("inf")
        y_min = float("inf")
        y_max = -float("inf")

        for pt in points:
            x = pt[0]
            y = pt[1]
            if x < x_min:
                x_min = x
            if x_max < x:
                x_max = x
            if y < y_min:
                y_min = y
            if y_max < y:
                y_max = y

        x_center = (x_min+x_max)/2
        y_center = (y_min+y_max)/2
        sg = (x_max-x_min)*(y_max-y_min)

        # 1.2 换算相对坐标
        for pt in points:
            pt[0]-=x_center
            pt[1]-=y_center
        
        return x_center, y_center, sg, points

    # 将坐标系分为 8 个区域，对区域编号
    def local_topo(self, pts0) :
        pts0 = pts0.tolist()
        pts_feat = []
        for pt in pts0: # 对于每个节点
            pt_feat = np.zeros(8) # 节点特征初始化
            for _pt in pts0:
                _pt[0] -= pt[0] # 换算相对坐标
                _pt[1] -= pt[1] # 换算相对坐标               
                x=_pt[0]
                y=_pt[1]
                if x==0 and y==0: 
                    continue
                if x > 0 and y > 0 and abs(x) > abs(y):
                    pt_feat[0]+=1
                elif x > 0 and y > 0 and abs(x) < abs(y):
                    pt_feat[1]+=1
                elif x < 0 and y > 0 and abs(x) < abs(y):
                    pt_feat[2]+=1        
                elif x < 0 and y > 0 and abs(x) > abs(y):
                    pt_feat[3]+=1
                elif x < 0 and y < 0 and abs(x) > abs(y):
                    pt_feat[4]+=1
                elif x < 0 and y < 0 and abs(x) < abs(y):
                    pt_feat[5]+=1
                elif x > 0 and y < 0 and abs(x) < abs(y):
                    pt_feat[6]+=1
                elif x > 0 and y < 0 and abs(x) > abs(y):
                    pt_feat[7]+=1

            pts_feat.append(pt_feat)

        return pts_feat
            
    def global_topo(self, points0, edge_indices):
        # 1.计算整图几何中心和面积 
        # 1.1 找整图的 box (x_min, x_max, y_min, y_max)
        # 1.2 换算相对坐标
        # pts0 = pts0.tolist()
        pts0 = points0.copy()
        x_center, y_center, sg, points = self._cal_box(pts0)

        # 三角剖分建边
        heads0 = edge_indices[0]
        tails0 = edge_indices[1]

        feats = []
        for pt_idx in range(pts0.shape[0]): # 从 0-9
            pt_arround = [] # 记录邻接点及本节点坐标
            pt_arround.append(pts0[pt_idx]) # 加入本节点坐标
            for idx, tail_idx in enumerate(tails0): # 对应每条边的尾节点序号
                if tail_idx==pt_idx:
                    pt_arround.append(pts0[heads0[idx]])

            # 2.找每个节点的子结构，计算子结构几何中心（极坐标）和面积
            x_acenter, y_acenter, sa, _ = self._cal_box(pt_arround)
            # 中心坐标转换为极坐标（直接使用转换公式）
            rho2 = x_acenter * x_acenter + y_acenter * y_acenter
            theta = math.atan2(y_acenter, x_acenter)
            # assert sg<1e-5, 'sg=0'
            s_r = sa/sg
            # 3.构成特征三元组：（A*的极坐标角度φ，A*的极坐标距离^2 ，子结构面积比 SA/SG）
            feat = [rho2, theta, s_r]
            feats.append(feat)

        return feats

    def convert_to_polar(self, points):
        # 转化为极坐标
        pts0 = points -128
        polar_pts = [cmath.polar(complex(i,j)) for i,j in pts0]

        return polar_pts

    def graph_topo(self, p_gt, divided_rho_coff=3,divided_phi_coff=8, size=256) :

        polar_pts = self.convert_to_polar(p_gt)
        y = np.zeros((divided_phi_coff, divided_rho_coff))

        for rho, phi in polar_pts:
            # 判断哪个区
            if phi <0:
                x1 = int((phi + math.pi * 2) // (2*math.pi/divided_phi_coff))
            else:
                x1 = int(phi // (2 * math.pi / divided_phi_coff))
            x2 = int(rho // (size/2*math.sqrt(2)/divided_rho_coff))
            y[x1,x2] += 1

        return y

    def set_num_graphs(self, num_graphs_in_matching_instance):
        self.num_graphs_in_matching_instance = num_graphs_in_matching_instance

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sampling_strategy = cfg.train_sampling if self.ds.sets == "train" else cfg.eval_sampling
        if self.num_graphs_in_matching_instance is None:
            raise ValueError("Num_graphs has to be set to an integer value.")

        idx = idx if self.true_epochs else None
        anno_list, perm_mat_list = self.ds.get_k_samples(idx, k=self.num_graphs_in_matching_instance, cls=self.cls, mode=sampling_strategy)

        for perm_mat in perm_mat_list:
            if (
                not perm_mat.size
                or (perm_mat.size < 2 * 2 and sampling_strategy == "intersection")
                and not self.true_epochs
            ):
                # 'and not self.true_epochs' because we assume all data is valid when sampling a true epoch
                next_idx = None if idx is None else idx + 1
                return self.__getitem__(next_idx)

        
        points_gt = [np.array([(kp["x"], kp["y"]) for kp in anno_dict["keypoints"]]) for anno_dict in anno_list]
        n_points_gt = [len(p_gt) for p_gt in points_gt]

        # # 点数不能少于 2，否则无法计算 box 和 sg
        if n_points_gt[0] <= 2 or n_points_gt[1] <= 2:
            next_idx = None if idx is None else idx + 1
            return self.__getitem__(next_idx)    

        graph_list = []
        loc_topo_feats = []
        glob_topo_feats = []
        graph_feats = []
        for p_gt, n_p_gt in zip(points_gt, n_points_gt):
            edge_indices, edge_features = build_graphs(p_gt, n_p_gt)
            
            loc_topo_feat = self.local_topo(p_gt)
            glob_topo_feat = self.global_topo(p_gt, edge_indices)
            graph_feat = self.graph_topo(p_gt)
            # Add dummy node features so the __slices__ of them is saved when creating a batch
            pos = torch.tensor(p_gt).to(torch.float32) / 256.0

            assert (pos > -1e-5).all(), p_gt
            graph = Data(
                edge_attr=torch.tensor(edge_features).to(torch.float32), # 边的向量
                edge_index=torch.tensor(edge_indices, dtype=torch.long),
                x=pos,
                pos=pos,
            )
            graph.num_nodes = n_p_gt
            graph_list.append(graph)
            loc_topo_feats.append(loc_topo_feat)
            glob_topo_feats.append(glob_topo_feat)
            graph_feats.append(graph_feat)
        ret_dict = {
            "Ps": [torch.Tensor(x) for x in points_gt],
            "ns": [torch.tensor(x) for x in n_points_gt],
            "gt_perm_mat": perm_mat_list,
            "edges": graph_list,
            "loc_topo_feats": [torch.Tensor(x) for x in loc_topo_feats],
            "glob_topo_feats": [torch.Tensor(x) for x in glob_topo_feats],
            "graph_feats": [torch.Tensor(x).view(-1) for x in graph_feats],
        }

        imgs = [anno["image"] for anno in anno_list]
        if imgs[0] is not None: # 走这里
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)])
            imgs = [trans(img) for img in imgs]
            ret_dict["images"] = imgs
        elif "feat" in anno_list[0]["keypoints"][0]:
            feat_list = [np.stack([kp["feat"] for kp in anno_dict["keypoints"]], axis=-1) for anno_dict in anno_list]
            ret_dict["features"] = [torch.Tensor(x) for x in feat_list]

        return ret_dict


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """

    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, "constant", 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        elif type(inp[0]) == Data:  # Graph from torch.geometric, create a batch
            ret = Batch.from_data_list(inp)
        else:
            raise ValueError("Cannot handle type {}".format(type(inp[0])))
        return ret

    ret = stack(data)
    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=False,
        worker_init_fn=worker_init_fix if fix_seed else worker_init_rand,
    )

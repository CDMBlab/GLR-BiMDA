import random
import torch as th
import numpy as np
import csv
import torch.utils.data.dataset as Dataset

def read_csv(path):
    """读取CSV文件"""
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        data = []
        data += [[float(i) for i in row] for row in reader]
        return th.Tensor(data)

def Simdata_pro(args):
    """只使用CSV数据预处理"""
    dataset = dict()
    
    # 从CSV文件加载数据
    mm_funsim = np.loadtxt(args.datapath + 'm_fs.csv', dtype=np.float64, delimiter=',')
    mm_seqsim = np.loadtxt(args.datapath + 'm_ss.csv', dtype=np.float64, delimiter=',')
    mm_gausim = np.loadtxt(args.datapath + 'm_gs.csv', dtype=np.float64, delimiter=',')
    dd_funsim = np.loadtxt(args.datapath + 'd_ts.csv', dtype=np.float64, delimiter=',')
    dd_semsim = np.loadtxt(args.datapath + 'd_ss.csv', dtype=np.float64, delimiter=',')
    dd_gausim = np.loadtxt(args.datapath + 'd_gs.csv', dtype=np.float64, delimiter=',')

    # 设置节点数量
    dataset['miRNA_number'] = mm_funsim.shape[0]
    dataset['disease_number'] = dd_funsim.shape[0]
    dataset['lncrna_number'] = 0  # 如果不使用lncRNA数据
    
    # 转换为PyTorch张量
    dataset['mm_f'] = th.FloatTensor(mm_funsim)
    dataset['mm_s'] = th.FloatTensor(mm_seqsim)
    dataset['mm_g'] = th.FloatTensor(mm_gausim)
    dataset['dd_t'] = th.FloatTensor(dd_funsim)
    dataset['dd_s'] = th.FloatTensor(dd_semsim)
    dataset['dd_g'] = th.FloatTensor(dd_gausim)

    # 计算平均相似度
    dataset['ms'] = (dataset['mm_f'] + dataset['mm_s'] + dataset['mm_g']) / 3
    dataset['ds'] = (dataset['dd_t'] + dataset['dd_s'] + dataset['dd_g']) / 3

    return dataset

def load_data(args):
    """加载关联数据"""
    # 加载miRNA-疾病关联矩阵
    md_matr = np.loadtxt(args.datapath + '/m_d.csv', dtype=np.float32, delimiter=',')
    
    # 创建训练测试分割
    rng = np.random.default_rng(seed=42)
    pos_samples = np.where(md_matr == 1)
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)
    
    neg_samples = np.where(md_matr == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]
    
    edge_idx_dict = dict()
    n_pos_samples = pos_samples_shuffled.shape[1]
    idx_split = int(n_pos_samples * 0.2)  # 80-20分割
    
    # 测试集
    test_pos_edges = pos_samples_shuffled[:, :idx_split]
    test_neg_edges = neg_samples_shuffled[:, :idx_split]
    test_pos_edges = test_pos_edges.T
    test_neg_edges = test_neg_edges.T
    test_true_label = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))
    test_true_label = np.array(test_true_label, dtype='float32')
    test_edges = np.vstack((test_pos_edges, test_neg_edges))
    
    # 训练集
    train_pos_edges = pos_samples_shuffled[:, idx_split:]
    train_neg_edges = neg_samples_shuffled[:, idx_split:]
    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T
    train_true_label = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0])))
    train_true_label = np.array(train_true_label, dtype='float32')
    train_edges = np.vstack((train_pos_edges, train_neg_edges))
    
    edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = train_true_label
    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = test_true_label
    
    # 加载关联类型
    md_class = np.loadtxt(args.datapath + '/m_d_edge.csv', dtype=np.float32, delimiter=',')
    edge_idx_dict['md_class'] = md_class
    edge_idx_dict['true_md'] = md_matr
    
    return edge_idx_dict

class EdgeDataset(Dataset.Dataset):
    def __init__(self, edges, labels):
        self.Data = edges
        self.Label = labels

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label
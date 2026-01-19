import torch
import numpy as np
import time
import gc
from utils import EdgeDataset
from GBiMDA import GLRBiMDA 
from sklearn import metrics
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
import os
import argparse


def get_metrics_binary(probs, labels, threshold=0.05):
    """二元分类指标计算"""
    # 大于0.05的都算有关联
    binary_pred = (probs > threshold).astype(int)
    
    accuracy = metrics.accuracy_score(labels, binary_pred)
    precision = metrics.precision_score(labels, binary_pred, zero_division=0)
    recall = metrics.recall_score(labels, binary_pred, zero_division=0)
    f1 = metrics.f1_score(labels, binary_pred, zero_division=0)
    
    # AUC
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    auc = metrics.auc(fpr, tpr)
    
    # AUPR
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(labels, probs)
    aupr = metrics.auc(recall_curve, precision_curve)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'aupr': aupr
    }

def train_test_new_model(simData, train_data, args, state):
    """新的训练测试函数"""
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    print("=" * 60)
    print("TRAINING NEW MODEL WITH PYTORCH RGCN")
    print("=" * 60)
    
    valid_ternary_metrics = []
    valid_binary_metrics = []
    
    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    test_edges = train_data['test_Edges']
    test_labels = train_data['test_Labels']
    m_d_matrix = train_data['true_md']
    md_class = train_data['md_class']
    
    n_pos = np.sum(train_labels == 1)
    ternary_train_labels = np.zeros_like(train_labels)
    ternary_test_labels = np.zeros_like(test_labels)
    
    pos_indices_train = np.where(train_labels == 1)[0]
    pos_indices_test = np.where(test_labels == 1)[0]
    
    np.random.seed(42)
    np.random.shuffle(pos_indices_train)
    np.random.shuffle(pos_indices_test)
    
    half_train = len(pos_indices_train) // 2
    half_test = len(pos_indices_test) // 2
    
    ternary_train_labels[pos_indices_train[:half_train]] = 1  # 上调
    ternary_train_labels[pos_indices_train[half_train:]] = 2  # 下调
    
    ternary_test_labels[pos_indices_test[:half_test]] = 1  # 上调
    ternary_test_labels[pos_indices_test[half_test:]] = 2  # 下调

    kfolds = args.kfold
    torch.manual_seed(42)

    if state == 'valid':
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
        train_idx, valid_idx = [], []

        for train_index, valid_index in kf.split(train_edges):
            train_idx.append(train_index)
            valid_idx.append(valid_index)

        for i in range(kfolds):
            print(f'\n{"="*50}')
            print(f'Fold {i + 1} of {kfolds} - New Model')
            print(f'{"="*50}')
            
            # 使用新模型
            model = GLRBiMDA(args).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
            labels_train_binary = train_labels[train_idx[i]]
            labels_valid_binary = train_labels[valid_idx[i]]
            labels_train_ternary = ternary_train_labels[train_idx[i]]
            labels_valid_ternary = ternary_train_labels[valid_idx[i]]
            
            trainEdges = EdgeDataset(edges_train, labels_train_ternary)
            validEdges = EdgeDataset(edges_valid, labels_valid_ternary)
            trainLoader = DataLoader.DataLoader(trainEdges, batch_size=args.batchSize, shuffle=True, num_workers=0)
            validLoader = DataLoader.DataLoader(validEdges, batch_size=args.batchSize, shuffle=True, num_workers=0)

            # 构建关联矩阵
            m_d_class = md_class.copy()
            m_d_class[tuple(edges_valid.T)] = 0.0
            md_m = torch.from_numpy(m_d_class).float().cuda()
            
            print("----- Training Phase -----")
            train_losses = []

            for e in range(args.epoch):
                running_loss = 0.0
                batch_count = 0
                
                model.train()
                start = time.time()
                
                for batch_idx, item in enumerate(trainLoader):
                    data, ternary_label = item
                    trainData = data.cuda()
                    trainLabel_ternary = ternary_label.cuda().long()
                    
                    # 前向传播
                    ternary_probs, binary_probs = model(simData, md_m, trainData)
                    
                    # 计算三元分类损失
                    ternary_loss = torch.nn.CrossEntropyLoss()(ternary_probs, trainLabel_ternary)
                    
                    # 计算二元分类损失
                    binary_labels = (trainLabel_ternary > 0).float()
                    binary_loss = torch.nn.BCELoss()(binary_probs, binary_labels)
                    
                    # 组合损失
                    loss = ternary_loss + binary_loss
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Batch {batch_idx+1}/{len(trainLoader)}, Loss: {loss.item():.4f}")
                
                end = time.time()
                epoch_loss = running_loss / batch_count
                train_losses.append(epoch_loss)
                
                print(f'Epoch {e+1} Summary:')
                print(f'  Time: {end-start:.2f}s, Loss: {epoch_loss:.4f}')

            # 验证阶段
            print("----- Validation Phase -----")
            valid_ternary_probs, valid_binary_probs = [], []
            valid_ternary_labels, valid_binary_labels = [], []
            
            model.eval()
            
            with torch.no_grad():
                for batch_idx, item in enumerate(validLoader):
                    data, ternary_label = item
                    validData = data.cuda()
                    
                    # 前向传播
                    ternary_probs_batch, binary_probs_batch = model(simData, md_m, validData)

                    valid_ternary_probs.append(ternary_probs_batch.cpu().numpy())
                    valid_binary_probs.append(binary_probs_batch.cpu().numpy())
                    valid_ternary_labels.append(ternary_label.numpy())
                    valid_binary_labels.append((ternary_label > 0).numpy())
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Validation Batch {batch_idx+1}/{len(validLoader)}")

                # 合并结果
                valid_ternary_probs = np.concatenate(valid_ternary_probs)
                valid_binary_probs = np.concatenate(valid_binary_probs)
                valid_ternary_labels = np.concatenate(valid_ternary_labels)
                valid_binary_labels = np.concatenate(valid_binary_labels)
                
                # 计算指标
                binary_metrics = get_metrics_binary(valid_binary_probs, valid_binary_labels)
                
                valid_binary_metrics.append(binary_metrics)
                
                print('Binary Classification:')
                print(f'  Acc: {binary_metrics["accuracy"]:.4f}, Prec: {binary_metrics["precision"]:.4f}')
                print(f'  Rec: {binary_metrics["recall"]:.4f}, F1: {binary_metrics["f1"]:.4f}')
                print(f'  AUC: {binary_metrics["auc"]:.4f}, AUPR: {binary_metrics["aupr"]:.4f}')
                
                # 清理GPU内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 交叉验证结果汇总
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS - NEW MODEL")
        print("="*60)
        
        # 二元分类结果
        print("\nBinary Classification Results:")
        binary_avg = {}
        for metric in valid_binary_metrics[0].keys():
            values = [fold[metric] for fold in valid_binary_metrics]
            avg_value = np.mean(values)
            std_value = np.std(values)
            binary_avg[metric] = (avg_value, std_value)
            print(f'{metric}: {avg_value:.4f} ± {std_value:.4f}')

    return  binary_avg

if __name__ == "__main__":
    from utils import Simdata_pro, load_data
    
    parser = argparse.ArgumentParser(description='New Model Training')
    
    # 训练参数
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--batchSize', type=int, default=64, help='batchSize')
    parser.add_argument('--Dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay')
    
    # 模型参数
    parser.add_argument('--kfold', type=int, default=5, help='kfold')
    parser.add_argument('--nei_size', type=list, default=[256, 32], help='nei_size')
    parser.add_argument('--hop', type=int, default=2, help='hop')
    parser.add_argument('--feture_size', type=int, default=256, help='feture_size')
    parser.add_argument('--edge_feature', type=int, default=9, help='edge_feature')
    parser.add_argument('--atthidden_fea', type=int, default=128, help='atthidden_fea')
    parser.add_argument('--sim_class', type=int, default=3, help='sim_class')
    parser.add_argument('--md_class', type=int, default=3, help='md_class')
    parser.add_argument('--m_num', type=int, default=853, help='m_num')
    parser.add_argument('--d_num', type=int, default=591, help='d_num')
    parser.add_argument('--view', type=int, default=3, help='view')
    
    args = parser.parse_args()
    args.datapath = './dataset/'
    args.data_dir = './dataset/'
    
    print("Loading data...")
    simData = Simdata_pro(args)
    train_data = load_data(args)
    
    print("Starting new model training...")
    ternary_results, binary_results = train_test_new_model(simData, train_data, args, 'valid')
    
    print("\n" + "="*60)
    print("NEW MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
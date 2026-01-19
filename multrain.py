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

def get_metrics_ternary(probs, labels, top_k=1):
    """三元分类指标计算"""
    # Top-1 准确率
    top1_pred = probs.argmax(axis=1)
    top1_accuracy = metrics.accuracy_score(labels, top1_pred)
    
    # Top-1 Precision, Recall, F1
    top1_precision = metrics.precision_score(labels, top1_pred, average='weighted', zero_division=0)
    top1_recall = metrics.recall_score(labels, top1_pred, average='weighted', zero_division=0)
    top1_f1 = metrics.f1_score(labels, top1_pred, average='weighted', zero_division=0)
    
    # 各类别的precision, recall, f1
    class_precision = metrics.precision_score(labels, top1_pred, average=None, zero_division=0)
    class_recall = metrics.recall_score(labels, top1_pred, average=None, zero_division=0)
    class_f1 = metrics.f1_score(labels, top1_pred, average=None, zero_division=0)
    
    return {
        'top1_accuracy': top1_accuracy,
        'top1_precision': top1_precision,
        'top1_recall': top1_recall,
        'top1_f1': top1_f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1
    }

def get_metrics_binary(probs, labels, threshold=0.5):
    binary_pred = (probs > threshold).astype(int)
    
    accuracy = metrics.accuracy_score(labels, binary_pred)
    precision = metrics.precision_score(labels, binary_pred, zero_division=0)
    recall = metrics.recall_score(labels, binary_pred, zero_division=0)
    f1 = metrics.f1_score(labels, binary_pred, zero_division=0)
    
    # 修复AUC计算问题
    try:
        # 检查标签是否只有一个类别
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            print(f"⚠️ 警告: 标签只有一个类别 {unique_labels[0]}，无法计算AUC")
            auc = 0.5  # 随机猜测的AUC
        else:
            fpr, tpr, _ = metrics.roc_curve(labels, probs)
            auc = metrics.auc(fpr, tpr)
    except Exception as e:
        print(f"⚠️ AUC计算错误: {e}")
        auc = 0.5
    
    # 修复AUPR计算问题
    try:
        # 检查正样本数量
        pos_count = np.sum(labels == 1)
        if pos_count == 0:
            print("⚠️ 警告: 没有正样本，无法计算AUPR")
            aupr = 0.0
        elif pos_count == len(labels):
            print("⚠️ 警告: 所有样本都是正样本，AUPR=1.0")
            aupr = 1.0
        else:
            precision_curve, recall_curve, _ = metrics.precision_recall_curve(labels, probs)
            aupr = metrics.auc(recall_curve, precision_curve)
    except Exception as e:
        print(f"⚠️ AUPR计算错误: {e}")
        aupr = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'aupr': aupr
    }

def get_labels_from_md_class(edges, md_class_matrix):
    """从md_class矩阵获取正确的标签"""
    ternary_labels = []
    binary_labels = []
    
    for edge in edges:
        m_idx, d_idx = int(edge[0]), int(edge[1])
        # md_class矩阵中的值: -1=上调, 1=下调, 2=其他, 0=无关联
        label_val = md_class_matrix[m_idx, d_idx]
        
        # 三元标签转换: 0=下调, 1=上调, 2=其他
        if label_val == -1:
            ternary_label = 1  # 上调 -> 1
        elif label_val == 1:
            ternary_label = 0  # 下调 -> 0
        elif label_val == 2:
            ternary_label = 2  # 其他 -> 2
        ternary_labels.append(ternary_label)
        
        if label_val in [-1, 1, 2]:  # 上调、下调、其他都是有关联
            binary_label = 1
        else:  # 0表示无关联
            binary_label = 0
        binary_labels.append(binary_label)
    
    return np.array(ternary_labels, dtype=np.int64), np.array(binary_labels, dtype=np.float32)

def train_test_new_model(simData, train_data, args, state):
    """新的训练测试函数"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 启用同步错误报告
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
    
    # ========== 修复：正确获取标签 ==========
    print("正在处理标签...")
    ternary_train_labels, binary_train_labels = get_labels_from_md_class(train_edges, md_class)
    ternary_test_labels, binary_test_labels = get_labels_from_md_class(test_edges, md_class)
    
    # 验证标签分布
    unique_train_ternary, counts_train_ternary = np.unique(ternary_train_labels, return_counts=True)
    
    ternary_label_names = {0: "下调", 1: "上调", 2: "其他"}
    

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
            labels_train_ternary = ternary_train_labels[train_idx[i]]
            labels_valid_ternary = ternary_train_labels[valid_idx[i]]
            labels_train_binary = binary_train_labels[train_idx[i]]
            labels_valid_binary = binary_train_labels[valid_idx[i]]
            
            # 验证标签范围
            print(f"训练集三元标签范围: {np.min(labels_train_ternary)} ~ {np.max(labels_train_ternary)}")
            print(f"验证集三元标签范围: {np.min(labels_valid_ternary)} ~ {np.max(labels_valid_ternary)}")
            
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
                    
                    # 获取对应的二元标签
                    batch_start = batch_idx * args.batchSize
                    batch_end = min((batch_idx + 1) * args.batchSize, len(labels_train_binary))
                    trainLabel_binary = torch.from_numpy(labels_train_binary[batch_start:batch_end]).float().cuda()
                    
                    # 验证标签范围
                    if torch.min(trainLabel_ternary) < 0 or torch.max(trainLabel_ternary) > 2:
                        print(f"❌ 错误: 三元标签超出范围! 最小值: {torch.min(trainLabel_ternary)}, 最大值: {torch.max(trainLabel_ternary)}")
                        continue
                    
                    
                    # 前向传播
                    ternary_probs, binary_probs = model(simData, md_m, trainData)
                    
                    # 计算三元分类损失
                    ternary_loss = torch.nn.CrossEntropyLoss()(ternary_probs, trainLabel_ternary)
        
                    binary_loss = torch.nn.BCELoss()(binary_probs.squeeze(), trainLabel_binary)
                    
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
                    batch_start = batch_idx * args.batchSize
                    batch_end = min((batch_idx + 1) * args.batchSize, len(labels_valid_binary))
                    batch_binary_labels = labels_valid_binary[batch_start:batch_end]
                    
                    # 前向传播
                    ternary_probs_batch, binary_probs_batch = model(simData, md_m, validData)

                    valid_ternary_probs.append(ternary_probs_batch.cpu().numpy())
                    valid_binary_probs.append(binary_probs_batch.cpu().numpy())
                    valid_ternary_labels.append(ternary_label.numpy())
                    valid_binary_labels.append(batch_binary_labels)
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Validation Batch {batch_idx+1}/{len(validLoader)}")

                # 合并结果
                valid_ternary_probs = np.concatenate(valid_ternary_probs)
                valid_binary_probs = np.concatenate(valid_binary_probs)
                valid_ternary_labels = np.concatenate(valid_ternary_labels)
                valid_binary_labels = np.concatenate(valid_binary_labels)
                
                # 计算指标
                ternary_metrics = get_metrics_ternary(valid_ternary_probs, valid_ternary_labels)
                binary_metrics = get_metrics_binary(valid_binary_probs.squeeze(), valid_binary_labels)
                
                valid_ternary_metrics.append(ternary_metrics)
                valid_binary_metrics.append(binary_metrics)
                
                print(f'Fold {i+1} Validation Results:')
            
                
                # 清理GPU内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 交叉验证结果汇总
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS - NEW MODEL")
        print("="*60)
        
        # 三元分类结果
        print("\n三元分类结果 (下调/上调/其他):")
        ternary_avg = {}
        for metric in ['top1_accuracy', 'top1_precision', 'top1_recall', 'top1_f1']:
            values = [fold[metric] for fold in valid_ternary_metrics]
            avg_value = np.mean(values)
            std_value = np.std(values)
            ternary_avg[metric] = (avg_value, std_value)
            print(f'{metric}: {avg_value:.4f} ')
    

    return ternary_avg

if __name__ == "__main__":
    from utils import Simdata_pro, load_data
    
    parser = argparse.ArgumentParser(description='New Model Training')
    
    # 训练参数
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--batchSize', type=int, default=64, help='batchSize')
    parser.add_argument('--Dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay')
    
    # 模型参数 - 保持三分类
    parser.add_argument('--kfold', type=int, default=5, help='kfold')
    parser.add_argument('--nei_size', type=list, default=[256, 32], help='nei_size')
    parser.add_argument('--hop', type=int, default=2, help='hop')
    parser.add_argument('--feture_size', type=int, default=256, help='feture_size')
    parser.add_argument('--edge_feature', type=int, default=9, help='edge_feature')
    parser.add_argument('--atthidden_fea', type=int, default=128, help='atthidden_fea')
    parser.add_argument('--sim_class', type=int, default=3, help='sim_class')
    parser.add_argument('--md_class', type=int, default=3, help='md_class')  # 保持三分类
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
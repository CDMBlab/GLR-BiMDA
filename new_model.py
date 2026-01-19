import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from otherlayers import CrossAttentionLayer, OnehotTran, NodeEmbedding, EdgeEmbedding, NeiAttention, NeiAggregator
from extractSubGraph import GetSubgraph

# ========== 创新点1: 负样本选择策略 ==========

class InternalNegativeSelector(nn.Module):
    """
    基于距离的负样本选择策略 - 创新点1
    """
    def __init__(self, feature_dim, max_iterations=30, convergence_threshold=1e-3):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
    def forward(self, positive_features, unlabeled_features, positive_mask):
        batch_size = positive_features.shape[0]
        
        if batch_size < 10 or positive_mask.sum() < 3 or (~positive_mask).sum() < 3:
            return torch.ones(batch_size, dtype=torch.bool, device=positive_features.device)
        
        # 分离正负样本
        positive_indices = torch.where(positive_mask)[0]
        negative_indices = torch.where(~positive_mask)[0]
        
        # 转换为numpy进行传统算法
        P_set = positive_features[positive_indices].detach().cpu().numpy()
        U_set = positive_features[negative_indices].detach().cpu().numpy()
        
        # MGCNSS负样本选择算法
        C_p = np.mean(P_set, axis=0) if len(P_set) > 0 else None
        C_u = np.mean(U_set, axis=0) if len(U_set) > 0 else None
        
        if C_p is None or C_u is None:
            return torch.ones(batch_size, dtype=torch.bool, device=positive_features.device)
        
        iteration = 0
        converged = False
        N_i_indices = list(range(len(U_set)))
        
        while iteration < self.max_iterations and not converged:
            C_p_old = C_p.copy()
            C_u_old = C_u.copy()
            
            # 使用余弦相似度划分
            P_i_indices = []
            N_i_indices = []
            
            for idx, sample in enumerate(U_set):
                CS_p = np.dot(sample, C_p) / (np.linalg.norm(sample) * np.linalg.norm(C_p) + 1e-8)
                CS_u = np.dot(sample, C_u) / (np.linalg.norm(sample) * np.linalg.norm(C_u) + 1e-8)
                
                if CS_p > CS_u:
                    P_i_indices.append(idx)
                else:
                    N_i_indices.append(idx)
            
            if len(P_i_indices) == 0 or len(N_i_indices) == 0:
                break
            
            # 更新质心
            P_i = U_set[P_i_indices]
            N_i = U_set[N_i_indices]
            C_p_new = np.mean(P_i, axis=0)
            C_u_new = np.mean(N_i, axis=0)
            
            # 使用欧氏距离细化划分
            P_i_refined_indices = []
            N_i_refined_indices = []
            
            for idx, sample in enumerate(U_set):
                ES_p = 1 / (1 + np.linalg.norm(sample - C_p_new))
                ES_u = 1 / (1 + np.linalg.norm(sample - C_u_new))
                
                if ES_p > ES_u:
                    P_i_refined_indices.append(idx)
                else:
                    N_i_refined_indices.append(idx)
            
            # 更新质心
            P_i_refined = U_set[P_i_refined_indices]
            N_i_refined = U_set[N_i_refined_indices]
            C_p = np.mean(P_i_refined, axis=0)
            C_u = np.mean(N_i_refined, axis=0)
            
            # 检查收敛
            diff_p = np.linalg.norm(C_p - C_p_old)
            diff_u = np.linalg.norm(C_u - C_u_old)
            
            if diff_p <= self.convergence_threshold and diff_u <= self.convergence_threshold:
                converged = True
            
            N_i_indices = N_i_refined_indices.copy()
            iteration += 1
        
        # 构建最终选择的样本掩码
        selected_mask = torch.zeros(batch_size, dtype=torch.bool, device=positive_features.device)
        selected_mask[positive_indices] = True
        
        reliable_negative_indices = negative_indices[N_i_indices]
        selected_mask[reliable_negative_indices] = True
        
        return selected_mask

# ========== 创新点2: 简化的元素级特征融合 ==========

class SimplifiedElementFusion(nn.Module):
    """
    简化的元素级特征融合 - 创新点2
    修复维度不匹配问题
    """
    def __init__(self, feature_dim, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 简单的特征融合权重
        self.fusion_weights = nn.Parameter(torch.ones(4))  # 4个特征源的权重
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, feature_list, query_features):
        """简化的元素级特征融合"""
        if len(feature_list) == 0:
            return query_features
        
        batch_size = query_features.shape[0]
        actual_components = min(len(feature_list), 4)
        feature_list = feature_list[:actual_components]
        
        # 堆叠特征 [batch_size, num_components, feature_dim]
        feature_tensor = torch.stack(feature_list, dim=1)
        
        # 简单的加权求和
        weights = F.softmax(self.fusion_weights[:actual_components], dim=0)
        weights = weights.view(1, actual_components, 1)  # [1, num_components, 1]
        
        # 加权求和 [batch_size, feature_dim]
        output = torch.sum(feature_tensor * weights, dim=1)
        
        return self.dropout(output)

# ========== 创新点3: 双通道双线性解码器 ==========
class EnhancedDualChannelDecoder(nn.Module):
    """
    双通道双线性解码器 - 创新点3
    修改为接受增强特征
    """
    
    def __init__(self, rgcn_dim=64, neighbor_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        self.rgcn_dim = rgcn_dim
        self.neighbor_dim = neighbor_dim  
        self.hidden_dim = hidden_dim
        self.multi_head = 8
        self.depth = int(hidden_dim / self.multi_head)
        
        # 双通道 Transformation Matrix
        self.TM = nn.Parameter(torch.empty(size=(2, self.multi_head, self.depth, self.depth)))
        nn.init.xavier_uniform_(self.TM.data, gain=1.414)
        
        # AATM矩阵
        self.AATM = nn.Parameter(torch.empty(size=(64, 64)))
        nn.init.xavier_uniform_(self.AATM.data, gain=0.5)
        
        # 特征对齐层 - 修改输入维度，现在输入是增强特征 + 邻居特征
        self.m_feature_align = nn.Sequential(
            nn.Linear(rgcn_dim + neighbor_dim, hidden_dim),  # rgcn_dim现在是增强特征的维度
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.d_feature_align = nn.Sequential(
            nn.Linear(rgcn_dim + neighbor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 交叉注意力层
        self.feature_interaction = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=self.multi_head,
            dropout=dropout,
            batch_first=True
        )
        
        # 简化的元素级特征融合
        self.element_fusion = SimplifiedElementFusion(
            feature_dim=hidden_dim,
            dropout=dropout
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64)
        )
        
        # 关系特征增强层
        self.relation_enhancer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, enhanced_m, enhanced_d, m_neighbor_emb, d_neighbor_emb, md_node):
        """
        修改后的前向传播，接受增强特征
        enhanced_m: [batch_size, rgcn_dim] - 交叉注意力增强的miRNA特征
        enhanced_d: [batch_size, rgcn_dim] - 交叉注意力增强的疾病特征
        """
        batch_size = md_node.shape[0]
        
        # 1. 处理邻居特征维度
        if len(m_neighbor_emb.shape) == 3:
            m_neighbor_emb = m_neighbor_emb.reshape(batch_size, -1)
        if len(d_neighbor_emb.shape) == 3:
            d_neighbor_emb = d_neighbor_emb.reshape(batch_size, -1)
        
        # 2. 特征拼接：增强特征 + 邻居特征
        m_combined = torch.cat([enhanced_m, m_neighbor_emb], dim=1)
        d_combined = torch.cat([enhanced_d, d_neighbor_emb], dim=1)
        
        # 3. 特征对齐
        m_aligned = self.m_feature_align(m_combined)
        d_aligned = self.d_feature_align(d_combined)
        
        # 4. 特征交互 - 多头注意力
        features = torch.stack([m_aligned, d_aligned], dim=1)
        attended_features, _ = self.feature_interaction(features, features, features)
        m_attended = attended_features[:, 0, :]
        d_attended = attended_features[:, 1, :]
        
        # 5. 简化的元素级特征融合
        feature_sources = [
            m_attended,  # 注意力增强的miRNA特征
            d_attended,  # 注意力增强的疾病特征
            m_aligned,   # 原始对齐的miRNA特征
            d_aligned    # 原始对齐的疾病特征
        ]
        
        m_fused = self.element_fusion(feature_sources, m_attended)
        d_fused = self.element_fusion(feature_sources, d_attended)
        
        # 6. 双通道 Transformation Matrix 处理
        m_fused_multi = m_fused.view(batch_size, self.multi_head, self.depth)
        d_fused_multi = d_fused.view(batch_size, self.multi_head, self.depth)
        
        # 双通道处理
        m_channels = torch.stack([F.gelu(m_fused_multi), m_fused_multi], dim=1)
        d_channels = torch.stack([F.gelu(d_fused_multi), d_fused_multi], dim=1)
        
        # 双通道矩阵乘法
        m_channels = m_channels.permute(1, 0, 2, 3)
        d_channels = d_channels.permute(1, 0, 2, 3)
        
        TM = self.TM
        
        # 通道1处理
        m_ch1 = m_channels[0]
        d_ch1 = d_channels[0]
        tm_ch1 = TM[0]
        
        # 通道2处理  
        m_ch2 = m_channels[1]
        d_ch2 = d_channels[1]
        tm_ch2 = TM[1]
        
        # 双通道双线性交互
        # 通道1: miRNA->疾病
        m2d_ch1 = torch.matmul(m_ch1.unsqueeze(2), tm_ch1.unsqueeze(0))
        m2d_ch1 = torch.matmul(m2d_ch1, d_ch1.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        
        # 通道2: miRNA->疾病  
        m2d_ch2 = torch.matmul(m_ch2.unsqueeze(2), tm_ch2.unsqueeze(0))
        m2d_ch2 = torch.matmul(m2d_ch2, d_ch2.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        
        # 合并双通道结果
        m2d_combined = torch.stack([m2d_ch1, m2d_ch2], dim=1)
        m2d_combined = torch.sum(m2d_combined, dim=1)
        m2d_combined = m2d_combined.view(batch_size, -1)
        
        # 7. 关系特征处理
        relation_features = torch.cat([m_fused, d_fused], dim=1)
        relation_logits = self.final_fusion(relation_features)
        
        # 双通道AATM处理
        relation_logits_ch1 = F.gelu(relation_logits)
        relation_logits_ch2 = relation_logits
        
        # 分别应用AATM变换
        relation_ch1 = torch.matmul(relation_logits_ch1, self.AATM)
        relation_ch2 = torch.matmul(relation_logits_ch2, self.AATM)
        
        # 合并双通道结果并增强
        relation_combined = relation_ch1 + relation_ch2
        relation_enhanced = self.relation_enhancer(relation_combined)
        
        # 最终融合
        final_features = torch.cat([m2d_combined, relation_enhanced], dim=1)
        final_features = self.dropout(final_features)
        
        return final_features

# ========== 基础组件保持不变 ==========

class SimMatrix(nn.Module):
    """相似度矩阵构建模块"""
    def __init__(self, args):
        super(SimMatrix, self).__init__()
        self.mnum = args.m_num
        self.dnum = args.d_num
        self.viewn = args.view
        from otherlayers import SimAttention
        self.attsim_m = SimAttention(self.mnum, self.mnum, self.viewn)
        self.attsim_d = SimAttention(self.dnum, self.dnum, self.viewn)

    def forward(self, data):
        m_funsim = data['mm_f'].cuda()
        m_seqsim = data['mm_s'].cuda()
        m_gossim = data['mm_g'].cuda()
        d_funsim = data['dd_t'].cuda()
        d_semsim = data['dd_s'].cuda()
        d_gossim = data['dd_g'].cuda()

        m_sim = torch.stack((m_funsim, m_seqsim, m_gossim), 0)
        d_sim = torch.stack((d_funsim, d_semsim, d_gossim), 0)
        m_attsim = self.attsim_m(m_sim)
        d_attsim = self.attsim_d(d_sim)

        m_final_sim = m_attsim.fill_diagonal_(fill_value=0)
        d_final_sim = d_attsim.fill_diagonal_(fill_value=0)

        return m_final_sim, d_final_sim

def build_ternary_association(m_d):
    """构建三元关联矩阵"""
    n_mirnas, n_diseases = m_d.shape
    ternary = torch.zeros((n_mirnas, n_diseases, 3), device=m_d.device)

    ternary[:, :, 0] = (m_d == -1).float()
    ternary[:, :, 1] = (m_d == 1).float()
    ternary[:, :, 2] = (m_d == 2).float()

    return ternary

class TrueRGCNConv(nn.Module):
    """真正的RGCN卷积层"""
    def __init__(self, in_dim, out_dim, num_relations, num_bases=None, dropout=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases if num_bases is not None else num_relations
        
        if self.num_bases > 0:
            self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
            self.coeff = nn.Parameter(torch.Tensor(num_relations, self.num_bases))
        else:
            self.relation_weights = nn.ParameterList([
                nn.Parameter(torch.Tensor(in_dim, out_dim)) for _ in range(num_relations)
            ])
        
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.basis_weights)
            nn.init.xavier_uniform_(self.coeff)
        else:
            for weight in self.relation_weights:
                nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.self_loop.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_type):
        device = x.device
        n_nodes = x.shape[0]
        
        self_message = self.self_loop(x)
        relation_messages = []
        
        if self.num_bases > 0:
            for r in range(self.num_relations):
                weight = torch.einsum('rb,bio->rio', self.coeff[r].unsqueeze(0), self.basis_weights)
                weight = weight.squeeze(0)
                
                mask = (edge_type == r)
                if mask.sum() == 0:
                    continue
                    
                r_edges = edge_index[:, mask]
                source_nodes = x[r_edges[0]]
                transformed = torch.mm(source_nodes, weight)
                relation_messages.append((r_edges[1], transformed))
        else:
            for r in range(self.num_relations):
                mask = (edge_type == r)
                if mask.sum() == 0:
                    continue
                    
                r_edges = edge_index[:, mask]
                source_nodes = x[r_edges[0]]
                transformed = torch.mm(source_nodes, self.relation_weights[r])
                relation_messages.append((r_edges[1], transformed))
        
        output = torch.zeros(n_nodes, self.out_dim, device=device)
        output = output + self_message
        
        for target_idx, message in relation_messages:
            output.index_add_(0, target_idx, message)
        
        output = output + self.bias
        output = self.dropout(output)
        return F.relu(output)

class TrueHeterogeneousRGCN(nn.Module):
    """真正的异质图RGCN实现"""
    def __init__(self, n_mirnas, n_diseases, embedding_dim=64, n_layers=2, dropout_rate=0.3):
        super().__init__()
        self.n_mirnas = n_mirnas
        self.n_diseases = n_diseases
        self.n_nodes = n_mirnas + n_diseases
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.num_relations = 8
        
        self.node_embeddings = nn.Parameter(torch.randn(self.n_nodes, embedding_dim))
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = TrueRGCNConv(
                in_dim=embedding_dim if i == 0 else embedding_dim,
                out_dim=embedding_dim,
                num_relations=self.num_relations,
                num_bases=4,
                dropout=dropout_rate
            )
            self.layers.append(layer)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_embeddings)
    
    def build_heterogeneous_edges(self, m_sim, d_sim, ternary_association):
        device = m_sim.device
        edges = []
        edge_types = []
        
        mm_edges = (m_sim > 0.3).nonzero(as_tuple=False).t()
        edges.append(mm_edges)
        edge_types.append(torch.zeros(mm_edges.shape[1], device=device, dtype=torch.long))
        
        dd_edges = (d_sim > 0.3).nonzero(as_tuple=False).t()
        dd_edges = dd_edges + self.n_mirnas
        edges.append(dd_edges)
        edge_types.append(torch.ones(dd_edges.shape[1], device=device, dtype=torch.long))
        
        for rel_idx in range(3):
            md_mask = (ternary_association[:, :, rel_idx] > 0.5)
            if md_mask.any():
                md_edges = md_mask.nonzero(as_tuple=False).t()
                md_edges[1] = md_edges[1] + self.n_mirnas
                edges.append(md_edges)
                edge_types.append(torch.full((md_edges.shape[1],), 2 + rel_idx, 
                                           device=device, dtype=torch.long))
        
        for rel_idx in range(3):
            dm_mask = (ternary_association[:, :, rel_idx] > 0.5)
            if dm_mask.any():
                dm_edges = dm_mask.nonzero(as_tuple=False).t()
                dm_edges = torch.stack([dm_edges[1] + self.n_mirnas, dm_edges[0]])
                edges.append(dm_edges)
                edge_types.append(torch.full((dm_edges.shape[1],), 5 + rel_idx, 
                                           device=device, dtype=torch.long))
        
        if edges:
            edge_index = torch.cat(edges, dim=1)
            edge_type = torch.cat(edge_types, dim=0)
        else:
            edge_index = torch.empty(2, 0, device=device, dtype=torch.long)
            edge_type = torch.empty(0, device=device, dtype=torch.long)
        
        return edge_index, edge_type
    
    def forward(self, m_sim, d_sim, ternary_association):
        device = m_sim.device
        x = self.node_embeddings.to(device)
        
        edge_index, edge_type = self.build_heterogeneous_edges(m_sim, d_sim, ternary_association)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        
        mirna_emb = x[:self.n_mirnas]
        disease_emb = x[self.n_mirnas:]
        
        return mirna_emb, disease_emb

# ========== 完整模型 ==========

class NewRGCNModel(nn.Module):
    """
    完整的新RGCN模型 - 集成6个创新点
    """
    def __init__(self, args):
        super().__init__()
        self.hop = getattr(args, 'hop', 2)
        self.neigh_size = getattr(args, 'nei_size', [256, 32])
        self.mNum = getattr(args, 'm_num', 853)
        self.dNum = getattr(args, 'd_num', 591)
        self.NodeFea = getattr(args, 'feture_size', 256)
        self.edgeFea = getattr(args, 'edge_feature', 9)
        self.drop = getattr(args, 'Dropout', 0.1)
        self.actfun = nn.LeakyReLU(0.2)

        # 基础组件
        self.SimGet = SimMatrix(args)
        self.edgeTran = OnehotTran(3, 3, self.mNum, self.dNum)
        self.getSubgraph = GetSubgraph(self.neigh_size, self.hop)
        self.EMBnode = NodeEmbedding(self.mNum, self.dNum, self.NodeFea, self.drop)
        self.EMBedge = EdgeEmbedding(3, 3, self.neigh_size)
        self.NeiAtt = NeiAttention(self.edgeFea, self.NodeFea, self.neigh_size)  # 创新点4: 邻居注意力
        self.Agg = NeiAggregator(self.NodeFea, self.drop, self.actfun)

        # 创新点5: 异质图RGCN
        self.rgcn = TrueHeterogeneousRGCN(n_mirnas=self.mNum, n_diseases=self.dNum, 
                                        embedding_dim=64, dropout_rate=0.3)
        
        # 交叉注意力层
        self.cross_attention = CrossAttentionLayer(emb_dim=64, dropout=self.drop)
        self.m_proj = nn.Linear(self.NodeFea, 64)
        self.d_proj = nn.Linear(self.NodeFea, 64)

        # 创新点3: 双通道双线性解码器 + 创新点2: 简化的元素级特征融合
        self.fusion_decoder = EnhancedDualChannelDecoder(
            rgcn_dim=64,
            neighbor_dim=64,
            hidden_dim=128,
            dropout=self.drop
        )

        # 分类器 - 动态调整输入维度
        decoder_output_dim = 8 + 64  # multi_head + relation_enhanced
        self.ternary_classifier = nn.Sequential(
            nn.Linear(decoder_output_dim, 32), nn.ReLU(), nn.Dropout(self.drop),
            nn.Linear(32, 3), nn.Softmax(dim=1)
        )
        self.binary_classifier = nn.Sequential(
            nn.Linear(decoder_output_dim, 32), nn.ReLU(), nn.Dropout(self.drop),
            nn.Linear(32, 1), nn.Sigmoid()
        )

        # 创新点1: 负样本选择策略
        self.negative_selector = InternalNegativeSelector(feature_dim=64)
        self.register_buffer('_selected_mask', None)
        
    def get_selected_mask(self):
        return self._selected_mask

    def forward(self, simData, m_d, md_node, training=True, labels=None):
        batch_size = md_node.shape[0]
        
        # 阶段1: 基础特征提取
        m_sim, d_sim = self.SimGet(simData)
        prep_one = torch.cat((m_sim, m_d), dim=1)
        prep_two = torch.cat((m_d.t(), d_sim), dim=1)
        md_all = torch.cat((prep_one, prep_two), dim=0)
        m_node = md_node[:, 0]
        d_node = md_node[:, 1] + self.mNum

        relation_adj = self.edgeTran(m_sim, d_sim, m_d)
        m_neinode_list, m_neirel_list, d_neinode_list, d_neirel_list = self.getSubgraph(
            m_node, d_node, md_all, relation_adj
        )
        m_nodeemb_list = self.EMBnode(m_sim, d_sim, m_neinode_list)
        d_nodeemb_list = self.EMBnode(m_sim, d_sim, d_neinode_list)
        m_relemb_list = self.EMBedge(m_neirel_list)
        d_relemb_list = self.EMBedge(d_neirel_list)

        # 阶段2: 邻居注意力聚合
        for i in range(self.hop - 1, 0, -1):
            mneigh_update_emb = self.NeiAtt(m_nodeemb_list[i], m_relemb_list[i], m_nodeemb_list[i + 1], i)
            dneigh_update_emb = self.NeiAtt(d_nodeemb_list[i], d_relemb_list[i], d_nodeemb_list[i + 1], i)
            m_nodeemb_list[i] = self.Agg(m_nodeemb_list[i], mneigh_update_emb)
            d_nodeemb_list[i] = self.Agg(d_nodeemb_list[i], dneigh_update_emb)

        m_neighbor_emb = m_nodeemb_list[0]
        d_neighbor_emb = d_nodeemb_list[0]

        # 阶段3: RGCN异质图学习
        ternary_assoc = build_ternary_association(m_d)
        rgcn_m_emb, rgcn_d_emb = self.rgcn(m_sim, d_sim, ternary_assoc)

        # 阶段4: 交叉注意力交互
        rgcn_m_batch = rgcn_m_emb[md_node[:, 0]]
        rgcn_d_batch = rgcn_d_emb[md_node[:, 1]]
        enhanced_m, enhanced_d = self.cross_attention(rgcn_m_batch.unsqueeze(0), rgcn_d_batch.unsqueeze(0))
        enhanced_m = enhanced_m.squeeze(0)
        enhanced_d = enhanced_d.squeeze(0)

        # 处理维度问题
        if len(m_neighbor_emb.shape) == 3:
            m_neighbor_emb = m_neighbor_emb.reshape(batch_size, -1)
        if len(d_neighbor_emb.shape) == 3:
            d_neighbor_emb = d_neighbor_emb.reshape(batch_size, -1)
        
        m_neighbor_proj = self.m_proj(m_neighbor_emb)
        d_neighbor_proj = self.d_proj(d_neighbor_emb)

        # 阶段5: 双通道双线性解码 + 简化的元素级特征融合
        fused_features = self.fusion_decoder(
            enhanced_m, enhanced_d, m_neighbor_proj, d_neighbor_proj, md_node
        )

        # 阶段6: 负样本选择
        if training and labels is not None:
            positive_mask = (labels == 1)
            selected_mask = self.negative_selector(fused_features, fused_features, positive_mask)
            self._selected_mask = selected_mask
        else:
            self._selected_mask = torch.ones(batch_size, dtype=torch.bool, device=fused_features.device)

        # 阶段7: 分类预测
        ternary_probs = self.ternary_classifier(fused_features)
        binary_probs = self.binary_classifier(fused_features).squeeze(1)

        return ternary_probs, binary_probs
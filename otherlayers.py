import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from dgl import nn as dglnn


class SimAttention(nn.Module):
    def __init__(self, num, feature, viewNum):
        super(SimAttention, self).__init__()

        self.Num = num
        self.FeaSize = feature
        self.viewn = viewNum
        self.dropout = nn.Dropout(0.3)
        self.fc_1 = nn.Linear(self.viewn, 150 * self.viewn, bias=False)
        self.fc_2 = nn.Linear(150 * self.viewn, self.viewn, bias=False)
        self.GAP1 = nn.AvgPool2d((self.Num, self.Num), (1, 1))

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, similarity):
        avr_pool = self.GAP1(similarity)

        sim_atten = avr_pool.reshape(-1, avr_pool.size(0))

        sim_atten = self.fc_1(sim_atten)
        sim_atten = self.relu(sim_atten)
        sim_atten = self.fc_2(sim_atten)

        sim_atten = sim_atten.reshape(similarity.size(0), 1, 1)
        all_att = self.softmax(sim_atten)
        sim = all_att * similarity

        final_sim = torch.sum(sim, dim=0, keepdim=False)

        return final_sim


class OnehotTran(nn.Module):
    def __init__(self, sim_class, md_class, m_num, d_num):
        super(OnehotTran, self).__init__()

        self.m_class = sim_class
        self.d_class = sim_class
        self.md_class = md_class
        self.class_all = self.m_class + self.d_class + self.md_class
        self.M_num = m_num
        self.D_num = d_num
        self.m_one = torch.ones(self.M_num, self.M_num)
        self.d_one = torch.ones(self.D_num, self.D_num)
        self.md_one = torch.ones(self.M_num, self.D_num)
        self.mone = self.m_one
        self.mtwo = self.mone + 1
        self.mthr = self.mtwo + 1
        self.done = self.d_one + self.d_class
        self.dtwo = self.done + 1
        self.dthr = self.dtwo + 1
        self.mdone = self.md_one + (self.md_class * 2)
        self.mdtwo = self.mdone + 1
        self.mdthr = self.mdtwo + 1

    def forward(self, m_score, d_score, md_score):
        mnew_score = torch.where(torch.ge(m_score, 0.65), self.mthr.cuda(), m_score)
        mnew_score = torch.where(torch.ge(mnew_score, 0.35) & torch.lt(mnew_score, 0.65), self.mtwo.cuda(), mnew_score)
        mnew_score = torch.where(torch.gt(mnew_score, 0.0) & torch.lt(mnew_score, 0.35), self.mone.cuda(), mnew_score)

        dnew_score = torch.where(torch.ge(d_score, 0.65), self.dthr.cuda(), d_score)
        dnew_score = torch.where(torch.ge(dnew_score, 0.35) & torch.lt(dnew_score, 0.65), self.dtwo.cuda(), dnew_score)
        dnew_score = torch.where(torch.gt(dnew_score, 0.0) & torch.lt(dnew_score, 0.35), self.done.cuda(), dnew_score)

        mdnew_score = torch.where(torch.eq(md_score, -1.0), self.mdone.cuda(), md_score)
        mdnew_score = torch.where(torch.eq(mdnew_score, 1.0), self.mdtwo.cuda(), mdnew_score)
        mdnew_score = torch.where(torch.eq(mdnew_score, 2.0), self.mdthr.cuda(), mdnew_score)

        pre_one = torch.cat((mnew_score, mdnew_score), dim=1)
        pre_two = torch.cat((mdnew_score.t(), dnew_score), dim=1)
        md_onepre = torch.cat((pre_one, pre_two), dim=0)

        return md_onepre


class NodeEmbedding(nn.Module):
    def __init__(self, m_num, d_num, feature, dropout):
        super(NodeEmbedding, self).__init__()

        self.m_Num = m_num
        self.d_Num = d_num
        self.node_voca_num = m_num + d_num
        self.fea_Size = feature
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(self.node_voca_num, self.fea_Size)
        self.relu = nn.ReLU()

    def forward(self, m_sim, d_sim, nei_node_list):
        batch_size = nei_node_list[0].size(0)
        md_f = torch.zeros(self.m_Num, self.d_Num).cuda()
        prep_m_f = torch.cat((m_sim, md_f), dim=1)
        prep_d_f = torch.cat((md_f.t(), d_sim), dim=1)
        m_d_f = torch.cat((prep_m_f, prep_d_f), dim=0)
        md_node_fea = self.drop(self.relu(self.linear(m_d_f)))

        neinode_emb_list = []
        for index in range(len(nei_node_list)):
            nei_node = nei_node_list[index]
            nei_node = torch.reshape(nei_node, (-1, 1)).squeeze(dim=1)
            nei_node_emb = torch.index_select(md_node_fea, 0, nei_node)
            neinode_emb_list.append(torch.reshape(nei_node_emb, (batch_size, -1, self.fea_Size)))

        return neinode_emb_list


class EdgeEmbedding(nn.Module):
    def __init__(self, sim_class, md_class, nei_size):
        super(EdgeEmbedding, self).__init__()

        self.m_class = sim_class
        self.d_class = sim_class
        self.md_class = md_class
        self.class_all = self.m_class + self.d_class + self.md_class
        self.neigh_size = nei_size
        self.bottom = torch.arange(start=0, end=self.class_all, step=1)
        self.bottom_onehot = torch.nn.functional.one_hot(self.bottom, self.class_all).float()

    def forward(self, nei_rel_list):
        batch_size = nei_rel_list[0].size(0)
        one_hot_emb = self.bottom_onehot.cuda()

        neirel_emb_list = []
        for index in range(len(nei_rel_list)):
            nei_relation = nei_rel_list[index]
            nei_relation = torch.reshape(nei_relation, (-1, 1)).squeeze(dim=1)
            nei_rel_emb = torch.index_select(one_hot_emb, 0, nei_relation)
            neirel_emb_list.append(torch.reshape(nei_rel_emb, (batch_size, -1, self.class_all)))

        return neirel_emb_list


class NeiAttention(nn.Module):
    def __init__(self, edgeFea, nodeFea, nei_size):
        super(NeiAttention, self).__init__()

        self.neigh_size = nei_size
        self.norm = 1 / sqrt(nodeFea)
        self.W1 = nn.Linear(edgeFea + nodeFea, nodeFea)
        self.actfun = nn.Softmax(dim=-1)

    def forward(self, x, x_nei_rel, x_nei_node, i):
        now_nei_size = self.neigh_size[i]
        n_neibor = int(int(x_nei_node.shape[1]) / now_nei_size)
        x = x.unsqueeze(dim=2)
        x_nei = torch.cat((x_nei_rel, x_nei_node), dim=-1)
        x_nei_up = self.W1(x_nei)
        x_nei_val = torch.reshape(x_nei_up, (x.shape[0], n_neibor, now_nei_size, -1))
        alpha = torch.matmul(x, x_nei_val.permute(0, 1, 3, 2)) * self.norm
        alpha = self.actfun(alpha)
        alpha = alpha.permute(0, 1, 3, 2)
        out = alpha * x_nei_val
        outputs = torch.sum(out, dim=2, keepdim=False)

        return outputs


class NeiAggregator(nn.Module):
    def __init__(self, nodeFea, dropout, actFunc, outBn=False, outAct=True, outDp=True):
        super(NeiAggregator, self).__init__()

        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(nodeFea)
        self.out = nn.Linear(nodeFea * 2, nodeFea)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x, x_nei):
        x_new = torch.cat((x, x_nei), dim=-1)
        x_new = self.out(x_new)

        if self.outBn:
            if len(x_new.shape) == 3:
                x_new = self.bns(x_new.transpose(1, 2)).transpose(1, 2)
            else:
                x_new = self.bns(x_new)
        if self.outAct: x_new = self.actFunc(x_new)
        if self.outDp: x_new = self.dropout(x_new)

        return x_new


class Attention(nn.Module):
    def __init__(self, edgeinSize, NodeinSize, outSize):
        super(Attention, self).__init__()

        self.edgeInSize = edgeinSize
        self.NodeInsize = NodeinSize
        self.outSize = outSize
        self.q = nn.Linear(self.edgeInSize, outSize)
        self.k = nn.Linear(self.NodeInsize + self.edgeInSize, outSize)
        self.v = nn.Linear(self.NodeInsize + self.edgeInSize, outSize)
        self.norm = 1 / sqrt(outSize)
        self.actfun1 = nn.Softmax(dim=-1)

    def forward(self, query, input):
        Q = self.q(query)
        K = self.k(input)
        V = self.v(input)
        alpha = torch.bmm(Q, K.permute(0, 2, 1)) * self.norm
        alpha = self.actfun1(alpha)
        z = (alpha.permute(0, 2, 1)) * V
        outputs = torch.sum(z, dim=1, keepdim=False)

        return outputs


class MLP(nn.Module):
    def __init__(self, inSize, outSize, dropout, actFunc, outBn=False, outAct=False, outDp=False):
        super(MLP, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        x = self.out(x)
        if self.outBn: x = self.bns(x)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x


# ========== 新增的RGCN相关层 ==========

class RelationalGraphConvolution(nn.Module):
    """关系图卷积层（修复版本）"""
    def __init__(self, input_dim, output_dim, num_relations, dropout=0.0):
        super(RelationalGraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.dropout = dropout
        
        # 为每种关系创建权重矩阵
        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_dim, output_dim))
            for _ in range(num_relations)
        ])
        
        # 偏置项
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, feat, adj_list):
        """
        Args:
            feat: 节点特征 [num_nodes, input_dim]
            adj_list: 邻接矩阵列表，每个adj形状为 [num_nodes, num_nodes]
        """
        # 对每种关系进行卷积
        outputs = []
        for i, adj in enumerate(adj_list):
            if i >= len(self.weights):  # 防止索引越界
                continue
                
            # 使用对应的权重矩阵
            transformed = torch.matmul(feat, self.weights[i])
            
            # 图卷积操作: A * X * W
            if adj is not None:
                # 确保adj是稠密矩阵
                if adj.is_sparse:
                    output = torch.spmm(adj, transformed)
                else:
                    output = torch.matmul(adj, transformed)
            else:
                output = transformed
            outputs.append(output)
        
        # 合并所有关系的输出
        if outputs:
            combined = torch.stack(outputs, dim=0).sum(dim=0)  # 使用sum而不是mean
        else:
            combined = torch.zeros_like(feat)
            
        # 添加偏置和应用激活函数
        output = combined + self.bias
        output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)
        
        return output


class MultiRelationRGCN(nn.Module):
    """多层RGCN"""
    def __init__(self, input_dim, hidden_dims, num_relations, dropout=0.1):
        super(MultiRelationRGCN, self).__init__()
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.layers.append(
                RelationalGraphConvolution(
                    dims[i], dims[i+1], num_relations, dropout
                )
            )
    
    def forward(self, feat, adj_list):
        for layer in self.layers:
            feat = layer(feat, adj_list)
        return feat


class CrossAttentionLayer(nn.Module):
    """交叉注意力层（PyTorch版本）"""
    def __init__(self, emb_dim, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.emb_dim = emb_dim
        
        # 投影矩阵
        self.query_projection = nn.Linear(emb_dim, emb_dim)
        self.key_projection = nn.Linear(emb_dim, emb_dim)
        self.value_projection = nn.Linear(emb_dim, emb_dim)
        self.output_projection = nn.Linear(emb_dim, emb_dim)
        
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, rna_embeddings, dis_embeddings):
        """
        Args:
            rna_embeddings: RNA节点嵌入 [batch_size, num_rna, emb_dim] 或 [num_rna, emb_dim]
            dis_embeddings: 疾病节点嵌入 [batch_size, num_dis, emb_dim] 或 [num_dis, emb_dim]
        """
        # 确保输入是3D张量 [batch_size, seq_len, emb_dim]
        if len(rna_embeddings.shape) == 2:
            rna_embeddings = rna_embeddings.unsqueeze(0)  # [1, num_rna, emb_dim]
        if len(dis_embeddings.shape) == 2:
            dis_embeddings = dis_embeddings.unsqueeze(0)  # [1, num_dis, emb_dim]
        
        batch_size, num_rna, emb_dim = rna_embeddings.shape
        _, num_dis, _ = dis_embeddings.shape
        
        # 投影查询、键、值
        rna_query = self.query_projection(rna_embeddings)  # [batch_size, num_rna, emb_dim]
        rna_key = self.key_projection(rna_embeddings)      # [batch_size, num_rna, emb_dim]
        rna_value = self.value_projection(rna_embeddings)  # [batch_size, num_rna, emb_dim]
        
        dis_query = self.query_projection(dis_embeddings)  # [batch_size, num_dis, emb_dim]
        dis_key = self.key_projection(dis_embeddings)      # [batch_size, num_dis, emb_dim]
        dis_value = self.value_projection(dis_embeddings)  # [batch_size, num_dis, emb_dim]
        
        # 计算注意力分数
        # rna_query: [batch_size, num_rna, emb_dim]
        # dis_key: [batch_size, num_dis, emb_dim] -> transpose: [batch_size, emb_dim, num_dis]
        rna_dis_scores = torch.matmul(rna_query, dis_key.transpose(1, 2))  # [batch_size, num_rna, num_dis]
        rna_dis_attn = F.softmax(rna_dis_scores, dim=-1)  # [batch_size, num_rna, num_dis]
        
        # dis_query: [batch_size, num_dis, emb_dim]
        # rna_key: [batch_size, num_rna, emb_dim] -> transpose: [batch_size, emb_dim, num_rna]
        dis_rna_scores = torch.matmul(dis_query, rna_key.transpose(1, 2))  # [batch_size, num_dis, num_rna]
        dis_rna_attn = F.softmax(dis_rna_scores, dim=-1)  # [batch_size, num_dis, num_rna]
        
        # 应用注意力权重
        rna_context = torch.matmul(rna_dis_attn, dis_value)  # [batch_size, num_rna, emb_dim]
        dis_context = torch.matmul(dis_rna_attn, rna_value)  # [batch_size, num_dis, emb_dim]
        
        # 残差连接 + 层归一化
        enhanced_rna = self.layer_norm(rna_embeddings + self.dropout(rna_context))
        enhanced_dis = self.layer_norm(dis_embeddings + self.dropout(dis_context))
        
        # 输出投影
        projected_rna = self.output_projection(enhanced_rna)
        projected_dis = self.output_projection(enhanced_dis)
        
        # 如果原始输入是2D，则返回2D
        if projected_rna.shape[0] == 1:
            projected_rna = projected_rna.squeeze(0)
            projected_dis = projected_dis.squeeze(0)
        
        return projected_rna, projected_dis


class InnerProductDecoder(nn.Module):
    """内积解码器"""
    def __init__(self, input_dim, num_relations, dropout=0.0):
        super(InnerProductDecoder, self).__init__()
        self.input_dim = input_dim
        self.num_relations = num_relations
        self.dropout = dropout
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: 节点嵌入 [num_nodes, input_dim]
        Returns:
            重构的邻接矩阵 [num_nodes, num_nodes]
        """
        reconstructions = torch.matmul(embeddings, embeddings.transpose(0, 1))
        reconstructions = torch.sigmoid(reconstructions)
        
        return reconstructions
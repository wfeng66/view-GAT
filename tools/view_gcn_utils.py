import torch
import torch.nn as nn
import torch.nn.functional as Functional
from torch_geometric.nn import GATConv, GCNConv

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn(nsample, xyz, new_xyz):
    dist = square_distance(xyz, new_xyz)
    id = torch.topk(dist,k=nsample,dim=1,largest=False)[1]
    id = torch.transpose(id, 1, 2)
    return id

class KNN_dist(nn.Module):
    def __init__(self,k):
        super(KNN_dist, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,1),
        )
        self.k=k
    def forward(self,F,vertices):
        # Ensure consistent tensor types
        F = F.to(dtype=torch.float)
        vertices = vertices.to(dtype=torch.float)
        
        id = knn(self.k, vertices, vertices)
        F = index_points(F,id)
        v = index_points(vertices,id)
        v_0 = v[:,:,0,:].unsqueeze(-2).repeat(1,1,self.k,1)
        v_F = torch.cat((v_0, v, v_0-v,torch.norm(v_0-v,dim=-1,p=2).unsqueeze(-1)),-1)
        v_F = self.R(v_F)
        F = torch.mul(v_F, F)
        F = torch.sum(F,-2)
        return F

class View_selector(nn.Module):
    def __init__(self, n_views, sampled_view, nclasses=40):
        super(View_selector, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.nclasses = nclasses
        self.cls = nn.Sequential(
            nn.Linear(512*self.s_views, 256*self.s_views),
            nn.LeakyReLU(0.2),
            nn.Linear(256*self.s_views, self.nclasses*self.s_views))
    def forward(self,F,vertices,k):
        # Ensure consistent tensor types
        F = F.to(dtype=torch.float)
        vertices = vertices.to(dtype=torch.float)
        
        id = farthest_point_sample(vertices,self.s_views)
        vertices1 = index_points(vertices,id)
        id_knn = knn(k,vertices,vertices1)
        F = index_points(F,id_knn)
        vertices = index_points(vertices,id_knn)
        F1 = F.transpose(1,2).reshape(F.shape[0],k,self.s_views*F.shape[-1])
        F_score = self.cls(F1).reshape(F.shape[0],k,self.s_views,self.nclasses).transpose(1,2)
        F1_ = Functional.softmax(F_score,-3)
        F1_ = torch.max(F1_,-1)[0]
        F1_id = torch.argmax(F1_,-1)
        F1_id = Functional.one_hot(F1_id,4).float()
        F1_id_v = F1_id.unsqueeze(-1).repeat(1,1,1,3)
        F1_id_F = F1_id.unsqueeze(-1).repeat(1, 1, 1, 512)
        F_new = torch.mul(F1_id_F,F).sum(-2)
        vertices_new = torch.mul(F1_id_v,vertices).sum(-2)
        return F_new,F_score,vertices_new

class LocalGCN(nn.Module):
    def __init__(self,k,n_views):
        super(LocalGCN,self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist(k=self.k)
    def forward(self,F,V):
        # Ensure consistent tensor types
        F = F.to(dtype=torch.float)
        V = V.to(dtype=torch.float)
        
        F = self.KNN(F, V)
        F = F.view(-1, 512)
        F = self.conv(F)
        F = F.view(-1, self.n_views, 512)
        return F

class NonLocalMP(nn.Module):
    def __init__(self,n_view):
        super(NonLocalMP,self).__init__()
        self.n_view=n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, F):
        # Ensure consistent tensor types
        F = F.to(dtype=torch.float)
        
        F_i = torch.unsqueeze(F, 2)
        F_j = torch.unsqueeze(F, 1)
        F_i = F_i.repeat(1, 1, self.n_view, 1)
        F_j = F_j.repeat(1, self.n_view, 1, 1)
        M = torch.cat((F_i, F_j), 3)
        M = self.Relation(M)
        M = torch.sum(M,-2)
        F = torch.cat((F, M), 2)
        F = F.view(-1, 512 * 2)
        F = self.Fusion(F)
        F = F.view(-1, self.n_view, 512)
        return F




class LocalGAT(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, heads=4, k=4):
        super(LocalGAT, self).__init__()
        self.k = k
        # one head, or multiple heads + concat
        self.gat = GATConv(in_channels, out_channels // heads, heads=heads, concat=True, dropout=0.2)

    def forward(self, F, V):
        # F: [B, N, C], V: [B, N, 3]
        B, N, C = F.size()
        # flatten batch:
        x = F.view(B*N, C)             # [B*N, C]
        pos = V.view(B*N, 3)           # [B*N, 3]
        # create batch vector so PyG knows sample boundaries:
        batch = torch.arange(B, device=F.device).view(-1,1).repeat(1,N).view(-1)
        # build edge_index via k‐NN on pos
        # PyG helper:
        from torch_geometric.nn import knn_graph
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)  # [2, E]

        # apply GAT
        out = self.gat(x, edge_index)  # [B*N, out_channels]
        # reshape back to [B, N, out_channels]
        return out.view(B, N, -1)



class GlobalGAT(nn.Module):
    """
    Global attention over all views: every node attends to every other.
    Supports different edge feature dimensions:
    - None: No edge features
    - 6: [v_i, v_j] - 6D edge features
    - 10: [v_i, v_j, v_i-v_j, |v_i-v_j|] - 10D edge features
    """
    def __init__(self, in_channels=512, out_channels=512, heads=4, edge_dim=10):
        super(GlobalGAT, self).__init__()
        # must divide evenly
        self.edge_dim = edge_dim
        if edge_dim is None:
            # No edge features
            self.gat = GATConv(in_channels, out_channels // heads, heads=heads, concat=True, dropout=0.2)
        else:
            # With edge features
            self.gat = GATConv(in_channels, out_channels // heads, heads=heads, concat=True, dropout=0.2, edge_dim=edge_dim)

    def forward(self, F, V, batch_size=None, return_attention=False):
        # F: [B, N, C] - node features
        # V: [B, N, 3] - camera positions (vertices)
        B, N, C = F.size()
        x = F.view(B*N, C)  # [B*N, C]

        # build a fully-connected edge_index per batch
        row, col = [], []
        for b in range(B):
            base = b * N
            idx = torch.arange(N, device=F.device)
            ii = idx.repeat_interleave(N) + base
            jj = idx.repeat(N)             + base
            row.append(ii);  col.append(jj)
        edge_index = torch.stack([torch.cat(row), torch.cat(col)], dim=0)

        if self.edge_dim is None:
            # No edge features
            if return_attention:
                out, (edge_index_out, attention_weights) = self.gat(x, edge_index, return_attention_weights=True)
            else:
                out = self.gat(x, edge_index)      # [B*N, out_channels]
        else:
            # Compute edge features based on edge_dim
            edge_attr = []
            for b in range(B):
                for i in range(N):
                    for j in range(N):
                        v_i = V[b, i]  # [3] - camera position i
                        v_j = V[b, j]  # [3] - camera position j
                        
                        if self.edge_dim == 6:
                            # 6D edge features: [v_i, v_j]
                            edge_feature = torch.cat([v_i, v_j], dim=0)  # [6]
                        elif self.edge_dim == 10:
                            # 10D edge features: [v_i, v_j, v_i-v_j, |v_i-v_j|]
                            v_diff = v_i - v_j  # [3] - difference vector
                            v_dist = torch.norm(v_diff, dim=0, keepdim=True)  # [1] - distance scalar
                            edge_feature = torch.cat([v_i, v_j, v_diff, v_dist], dim=0)  # [10]
                        else:
                            raise ValueError(f"Unsupported edge_dim: {self.edge_dim}. Must be None, 6, or 10.")
                        
                        edge_attr.append(edge_feature)
            edge_attr = torch.stack(edge_attr, dim=0)  # [B*N*N, edge_dim]
            
            if return_attention:
                out, (edge_index_out, attention_weights) = self.gat(x, edge_index, edge_attr, return_attention_weights=True)
            else:
                out = self.gat(x, edge_index, edge_attr)      # [B*N, out_channels]
        
        if return_attention:
            # Reshape attention weights to [B, N, N] format
            # attention_weights shape: [num_edges, heads] -> [B, N, N, heads] -> [B, N, N] (mean over heads)
            # attention_weights correspond to edges in edge_index
            num_edges = attention_weights.shape[0]
            num_heads = attention_weights.shape[1]
            
            # Create attention matrix [B, N, N]
            attention_matrix = torch.zeros(B, N, N, device=x.device, dtype=attention_weights.dtype)
            
            # Fill the attention matrix using edge_index and attention_weights
            for i in range(num_edges):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                batch_idx = src // N
                src_idx = src % N
                dst_idx = dst % N
                
                # Average attention across heads
                attention_matrix[batch_idx, src_idx, dst_idx] = attention_weights[i].mean()
            
            return out.view(B, N, -1), attention_matrix
        else:
            return out.view(B, N, -1)          # → [B, N, out_channels]


class GlobalGCN(nn.Module):
    """
    Global GCN over all views: fully-connected graph where every node connects to every other.
    Uses standard GCN convolution instead of attention-based GAT.
    """
    def __init__(self, in_channels=512, out_channels=512, heads=4, edge_dim=None):
        super(GlobalGCN, self).__init__()
        # Note: GCN doesn't use heads or edge_dim, but we keep them for API compatibility
        # We use multiple GCN layers to match the capacity of multi-head GAT
        self.num_layers = heads  # Use 'heads' as number of parallel GCN layers for fair comparison
        self.edge_dim = edge_dim  # Stored but not used (for API compatibility)
        
        # Multiple GCN layers whose outputs are concatenated (similar to multi-head attention)
        self.gcn_layers = nn.ModuleList([
            GCNConv(in_channels, out_channels // heads)
            for _ in range(heads)
        ])
        
        # Optional: batch normalization after GCN
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, F, V, batch_size=None, return_attention=False):
        # F: [B, N, C] - node features
        # V: [B, N, 3] - camera positions (vertices) - not used in GCN but kept for API compatibility
        B, N, C = F.size()
        x = F.view(B*N, C)  # [B*N, C]

        # build a fully-connected edge_index per batch
        row, col = [], []
        for b in range(B):
            base = b * N
            idx = torch.arange(N, device=F.device)
            ii = idx.repeat_interleave(N) + base
            jj = idx.repeat(N)             + base
            row.append(ii);  col.append(jj)
        edge_index = torch.stack([torch.cat(row), torch.cat(col)], dim=0)

        # Apply multiple GCN layers and concatenate outputs (like multi-head attention)
        out_list = []
        for gcn in self.gcn_layers:
            out_list.append(gcn(x, edge_index))
        out = torch.cat(out_list, dim=-1)  # [B*N, out_channels]
        
        # Apply batch normalization
        out = self.bn(out)
        
        if return_attention:
            # GCN doesn't have attention weights, return uniform weights for compatibility
            # Create a uniform attention matrix [B, N, N] where each node equally attends to all
            attention_matrix = torch.ones(B, N, N, device=x.device, dtype=out.dtype) / N
            return out.view(B, N, -1), attention_matrix
        else:
            return out.view(B, N, -1)          # → [B, N, out_channels]


class AttentionPooling(nn.Module):
    """
    Learned attention pooling head for global descriptor formation.
    Uses attention weights to pool over views.
    """
    def __init__(self, in_channels=512, hidden_dim=256, num_heads=8):
        super(AttentionPooling, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, in_channels))
        
        # Optional: additional processing layers
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, in_channels)
        )
        
    def forward(self, x):
        # x: [B, N, C] where B=batch, N=num_views, C=channels
        B, N, C = x.shape
        
        # Create query for each batch
        query = self.query.expand(B, -1, -1)  # [B, 1, C]
        
        # Apply multi-head attention
        attn_out, attn_weights = self.attention(
            query, x, x  # query, key, value
        )  # attn_out: [B, 1, C], attn_weights: [B, 1, N]
        
        # Optional projection
        pooled = self.proj(attn_out.squeeze(1))  # [B, C]
        
        return pooled, attn_weights.squeeze(1)  # [B, C], [B, N]


# class SaliencyScorer(nn.Module):
#     """
#     Attention-based saliency scorer for node selection.
#     Scores each node by aggregated incoming attention or learned features.
#     """
#     def __init__(self, in_channels=512, hidden_dim=128, att_lambda=0.3):
#         super(SaliencyScorer, self).__init__()
#         self.in_channels = in_channels
#         self.hidden_dim = hidden_dim
#         self.att_lambda = att_lambda  # Weight for attention coefficients
        
#         # Small head on GAT features for saliency scoring
#         self.saliency_head = nn.Sequential(
#             nn.Linear(in_channels, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()  # Output saliency scores between 0 and 1
#         )
        
#     def forward(self, x, attention_weights=None):
#         # x: [B, N, C] - GAT features
#         # attention_weights: [B, N, N] - attention weights from GAT (optional)
#         B, N, C = x.shape
        
#         # Method 1: Use learned features
#         saliency_scores = self.saliency_head(x)  # [B, N, 1]
#         saliency_scores = saliency_scores.squeeze(-1)  # [B, N]
        
#         # Method 2: If attention weights are provided, use aggregated incoming attention
#         if attention_weights is not None:
#             # Aggregate incoming attention for each node
#             incoming_attention = attention_weights.sum(dim=1)  # [B, N]
#             # Combine with learned scores using att_lambda for attention coefficients
#             saliency_scores = (1.0 - self.att_lambda) * saliency_scores + self.att_lambda * incoming_attention
        
#         return saliency_scores  # [B, N]


class SaliencyScorer(nn.Module):
    """
    Returns (logits, probs) where probs∈[0,1].
    Attention is blended in PROB space via att_lambda (non-trainable).
    """
    def __init__(self, in_channels=512, hidden_dim=128, att_lambda: float = 0.0):
        super().__init__()
        self.att_lambda = float(att_lambda)                  # <- NOT a Parameter
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)                   # logits per node
        )

    def forward(self, x, attention_weights=None):
        # x: [B,N,C], attention_weights (optional): [B,N,N]
        logits = self.mlp(x).squeeze(-1)                    # [B,N]
        probs  = torch.sigmoid(logits)                      # [B,N]
        if attention_weights is not None and self.att_lambda > 0:
            incoming = attention_weights.sum(dim=1)         # [B,N]
            incoming = incoming / (incoming.sum(dim=1, keepdim=True) + 1e-8)
            probs = (1.0 - self.att_lambda) * probs + self.att_lambda * incoming
            # keep logits consistent with probs
            logits = torch.log(probs.clamp_min(1e-8)) - torch.log1p(-probs.clamp_max(1 - 1e-8))
        return logits, probs




# class TopKSelector(nn.Module):
#     """
#     Select top-k nodes based on saliency scores.
#     """
#     def __init__(self, k=4, differentiable=False, temperature=1.0):
#         super(TopKSelector, self).__init__()
#         self.k = k
#         self.differentiable = differentiable
#         self.temperature = temperature
        
#     def forward(self, features, saliency_scores):
#         # features: [B, N, C]
#         # saliency_scores: [B, N]
#         B, N, C = features.shape
        
#         if self.differentiable:
#             return self._differentiable_topk(features, saliency_scores)
#         else:
#             return self._hard_topk(features, saliency_scores)
    
#     def _hard_topk(self, features, saliency_scores):
#         """Hard top-k selection (non-differentiable)"""
#         B, N, C = features.shape
#         # Get top-k indices
#         _, topk_indices = torch.topk(saliency_scores, k=min(self.k, N), dim=1)  # [B, k]
        
#         # Select features based on top-k indices
#         batch_indices = torch.arange(B, device=features.device).unsqueeze(1)  # [B, 1]
#         selected_features = features[batch_indices, topk_indices]  # [B, k, C]
        
#         return selected_features, topk_indices
    
#     def _differentiable_topk(self, features, saliency_scores):
#         """Differentiable top-k selection using Gumbel-Softmax"""
#         B, N, C = features.shape
#         k = min(self.k, N)
        
#         # Apply Gumbel-Softmax to make selection differentiable
#         # Add small epsilon to avoid log(0)
#         eps = 1e-8
#         log_scores = torch.log(saliency_scores + eps)  # [B, N]
        
#         # Sample Gumbel noise
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_scores) + eps) + eps)
        
#         # Add Gumbel noise and apply temperature
#         gumbel_scores = (log_scores + gumbel_noise) / self.temperature  # [B, N]
        
#         # Apply softmax to get soft selection weights
#         soft_weights = torch.softmax(gumbel_scores, dim=1)  # [B, N]
        
#         # Get top-k soft weights
#         _, topk_indices = torch.topk(soft_weights, k=k, dim=1)  # [B, k]
        
#         # Create one-hot-like weights for top-k selection
#         topk_weights = torch.zeros_like(soft_weights)  # [B, N]
#         batch_indices = torch.arange(B, device=features.device).unsqueeze(1)  # [B, 1]
#         topk_weights[batch_indices, topk_indices] = soft_weights[batch_indices, topk_indices]
        
#         # Normalize weights to sum to 1 for each batch
#         topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + eps)  # [B, N]
        
#         # Weighted sum of features
#         selected_features = torch.bmm(topk_weights.unsqueeze(1), features)  # [B, 1, C]
#         selected_features = selected_features.squeeze(1)  # [B, C]
        
#         # Repeat to get [B, k, C] shape for consistency
#         selected_features = selected_features.unsqueeze(1).repeat(1, k, 1)  # [B, k, C]
        
#         return selected_features, topk_indices



def neural_sort(scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    NeuralSort: Differentiable sorting operator from https://arxiv.org/pdf/1903.08850
    
    Computes a soft permutation matrix P where P[i, j] represents the probability
    that element j has rank i (i.e., is the i-th largest element).
    
    Args:
        scores: [B, N] tensor of scores (higher = more likely to be top-ranked)
        tau: temperature parameter (lower = sharper, approaches hard sort as tau -> 0)
    
    Returns:
        P: [B, N, N] soft permutation matrix where:
           - Row i represents the soft selection for rank i (1st largest, 2nd largest, etc.)
           - Each row sums to 1 (probability distribution over positions)
           - In the limit tau -> 0, P becomes a hard permutation matrix
    """
    B, V = scores.shape
    device, dtype = scores.device, scores.dtype

    s   = scores
    s_i = s.unsqueeze(2)                 # [B,V,1]
    s_j = s.unsqueeze(1)                 # [B,1,V]
    A   = (s_i - s_j).abs()              # [B,V,V]
    ones = torch.ones(V, 1, device=device, dtype=dtype)  # [V,1]
    A1   = torch.matmul(A, ones).squeeze(-1)             # [B,V]

    ranks = torch.arange(1, V+1, device=device, dtype=dtype)  # [V]
    scale = (V + 1 - 2 * ranks).view(1, V, 1)                 # [1,V,1]

    logits = scale * s.unsqueeze(1) - A1.unsqueeze(1)   # [B,V,V]
    P = torch.softmax(logits / (tau + 1e-10), dim=-1)   # row-stochastic
    return P


class TopKSelector(nn.Module):
    """
    Differentiable Top-K selector using NeuralSort.
    
    Uses soft permutation matrices to enable gradient flow through top-k selection.
    Based on: "Stochastic Optimization of Sorting Networks via Continuous Relaxations"
    https://arxiv.org/pdf/1903.08850
    
    When straight_through=True (--diff_topk flag):
        - Forward: Uses soft permutation rows (convex combinations)
        - Backward: Gradients flow through the soft permutation
    
    When straight_through=False:
        - Uses hard selection (standard top-k with no gradient through selection)
    """
    
    def __init__(self, temperature: float = 1.0, logit_tau: float = 1.0, straight_through: bool = True):
        super().__init__()
        self.register_buffer('temperature', torch.tensor(float(temperature)))
        self.register_buffer('logit_tau', torch.tensor(float(logit_tau)))  # kept for API compatibility
        self.straight_through = straight_through

    def set_temperature(self, tau: float | None = None, logit_tau: float | None = None):
        """Update temperature parameters (for annealing during training)."""
        if tau is not None:
            self.temperature.fill_(float(tau))
        if logit_tau is not None:
            self.logit_tau.fill_(float(logit_tau))

    def forward(self, features: torch.Tensor, logits: torch.Tensor, k: int):
        """
        Select top-k features using NeuralSort-based soft permutation.
        
        Args:
            features: [B, V, C] - node features
            logits: [B, V] - saliency scores (higher = more important)
            k: number of nodes to select
        
        Returns:
            sel_feats: [B, K, C] - selected features
            idx: [B, K] - hard top-k indices (for logging/visualization)
            sel_w: [B, K] - selection weights for selected nodes
            mask_all: [B, V] - soft mask over all views (sum to 1)
            M: [B, K, V] - selection matrix (each row selects one output node)
        """
        B, V, C = features.shape
        device = features.device
        dtype = features.dtype

        # Normalize logits for numerical stability
        logits_norm = (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True).clamp_min(1e-6))

        # Get hard top-k indices (always needed for reference and eval)
        _, idx = torch.topk(logits_norm, k, dim=1)  # [B, K]
        
        # Create hard selection matrix H
        H = torch.zeros(B, k, V, device=device, dtype=dtype)
        H.scatter_(2, idx.unsqueeze(-1), 1.0)  # [B, K, V] - one-hot rows
        
        # Compute soft weights for all views (simple softmax for mask_all)
        mask_all = torch.softmax(logits_norm / (self.temperature + 1e-10), dim=1)  # [B, V]
        
        # Selection weights for the k selected nodes
        sel_w = torch.gather(mask_all, 1, idx)  # [B, K]

        if not self.training:
            # Evaluation mode: always use hard selection
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, K, C]
            sel_feats = features.gather(1, idx_exp)  # [B, K, C]
            M = H
            return sel_feats, idx, sel_w, mask_all, M

        # Training mode
        if not self.straight_through:
            # Hard selection (no gradient through selection)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, K, C]
            sel_feats = features.gather(1, idx_exp)  # [B, K, C]
            M = H
        else:
            # NeuralSort-based soft selection
            # Compute soft permutation matrix
            P = neural_sort(logits_norm, tau=float(self.temperature))  # [B, V, V]
            
            # Take first k rows of P as selector M
            # Row i of P gives the soft selection for the i-th highest scoring element
            M_soft = P[:, :k, :]  # [B, K, V]
            
            # Straight-through: forward uses hard, backward uses soft
            # M = H + (M_soft - H).detach() would give hard forward, soft backward
            # But we want soft forward for better gradient flow during training
            # Use: forward = soft, backward = soft (pure differentiable)
            M = M_soft  # [B, K, V]
            
            # Compute selected features using soft selection
            sel_feats = torch.bmm(M, features)  # [B, K, C]

        return sel_feats, idx, sel_w, mask_all, M





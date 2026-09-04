import numpy as np
import torch, math
import torch.nn as nn
import torchvision.models as models
from .Model import Model
from tools.view_gcn_utils import GlobalGAT, GlobalGCN, AttentionPooling, SaliencyScorer, TopKSelector

mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='resnet18', dataset='modelnet40'):
        super(SVCNN, self).__init__(name)
        if dataset == 'modelnet40':
            self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                               'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                               'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                               'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        elif dataset == 'scanobjectnn':
            self.classnames = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 
                              'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']
        elif dataset == 'colombia':
            self.classnames = ['0', '1', '2', '3', '4', '5']
        # self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.use_vit = cnn_name.startswith('vit')
        self.use_densenet = cnn_name.startswith('densenet')
        self.mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
            elif self.cnn_name == 'resnet101':
                self.net = models.resnet101(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
            elif self.cnn_name == 'resnet152':
                self.net = models.resnet152(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
            else:
                raise ValueError(f"Unsupported ResNet variant: {self.cnn_name}")
        
        elif self.use_densenet:
            if self.cnn_name == 'densenet121':
                self.net = models.densenet121(pretrained=self.pretraining)
                self.net.classifier = nn.Linear(1024, self.nclasses)
            elif self.cnn_name == 'densenet161':
                self.net = models.densenet161(pretrained=self.pretraining)
                self.net.classifier = nn.Linear(2208, self.nclasses)
            elif self.cnn_name == 'densenet169':
                self.net = models.densenet169(pretrained=self.pretraining)
                self.net.classifier = nn.Linear(1664, self.nclasses)
            elif self.cnn_name == 'densenet201':
                self.net = models.densenet201(pretrained=self.pretraining)
                self.net.classifier = nn.Linear(1920, self.nclasses)
            else:
                raise ValueError(f"Unsupported DenseNet variant: {self.cnn_name}")
        
        elif self.use_vit:
            if self.pretraining:
                weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.net = models.vit_b_16(weights=weights)
            self.net.heads.head = nn.Linear(self.net.heads.head.in_features, self.nclasses)
            
            # Store ViT-specific transforms for proper preprocessing
            if self.pretraining:
                self.vit_transforms = weights.transforms(antialias=True)
            else:
                # Fallback transforms for non-pretrained ViT
                self.vit_transforms = None

        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11_bn(pretrained=self.pretraining).features
                self.net_2 = models.vgg11_bn(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)

    def freeze_backbone(self):
        """Freeze backbone parameters (all except classifier head)"""
        if self.use_resnet:
            # Freeze all layers except fc
            for name, param in self.net.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
        elif self.use_densenet:
            # Freeze all layers except classifier
            for name, param in self.net.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        elif self.use_vit:
            # Freeze all layers except heads.head
            for name, param in self.net.named_parameters():
                if 'heads.head' not in name:
                    param.requires_grad = False
        else:
            # For AlexNet/VGG: freeze net_1 (features), keep net_2 (classifier) trainable
            for param in self.net_1.parameters():
                param.requires_grad = False
        self.train()  # Keep in train mode, just freeze params
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters (allow full fine-tuning)"""
        if hasattr(self, 'net'):
            for param in self.net.parameters():
                param.requires_grad = True
        else:
            # For AlexNet/VGG: unfreeze net_1
            for param in self.net_1.parameters():
                param.requires_grad = True

    def forward(self, x):
        if self.use_vit:
            if x.shape[1] == 1:  # Convert grayscale to 3-channel for ViT
                x = x.repeat(1, 3, 1, 1)
            return self.net(x)
        elif self.use_resnet or self.use_densenet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0], -1))
        

class view_GCN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='resnet18', num_views=20, dataset='modelnet40', edge_dim=10, att_lambda=0.3, diff_topk=False, n_attn_heads=None, num_levels=3, graph_net='gat', **kwargs):
        super(view_GCN, self).__init__(name)
        # Handle num_levels from kwargs if not provided directly (for backward compatibility)
        self.num_levels = kwargs.get('num_levels', num_levels)
        # Handle graph_net from kwargs if not provided directly (for backward compatibility)
        self.graph_net = kwargs.get('graph_net', graph_net).lower()
        
        # Handle n_attn_heads from kwargs if not provided directly (for backward compatibility)
        if n_attn_heads is None:
            n_attn_heads = kwargs.get('n_attn_heads', None)
        # Set default attention heads if not provided
        if n_attn_heads is None:
            n_attn_heads = [8] * self.num_levels  # Default: 8 heads for all layers
        # If single value, replicate for all levels
        if len(n_attn_heads) == 1:
            n_attn_heads = n_attn_heads * self.num_levels
        if len(n_attn_heads) != self.num_levels:
            raise ValueError(f"n_attn_heads must have exactly {self.num_levels} values, got {len(n_attn_heads)}")
        self.n_attn_heads = n_attn_heads
        # self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
        #                    'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
        #                    'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
        #                    'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
        #                    'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        # self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        if dataset == 'modelnet40':
            self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                                'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                                'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                                'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                                'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        elif dataset == 'scanobjectnn':
            self.classnames = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 
                                'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']
        elif dataset == 'colombia':
            self.classnames = ['0', '1', '2', '3', '4', '5']
        self.diff_topk = bool(diff_topk)
        self.nclasses = nclasses
        self.num_views = num_views
        self.cnn_name = cnn_name
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, requires_grad=False)
        self.use_resnet = cnn_name.startswith('resnet')
        self.use_vit = cnn_name.startswith('vit')
        self.use_densenet = cnn_name.startswith('densenet')
        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
            # Determine feature dimension based on CNN backbone
            if cnn_name in ['resnet18', 'resnet34']:
                self.feature_dim = 512
            elif cnn_name in ['resnet50', 'resnet101', 'resnet152']:
                self.feature_dim = 2048
            else:
                self.feature_dim = 512  # default fallback
        elif self.use_densenet:
            # For DenseNet, extract features (everything except classifier)
            # DenseNet structure: features -> ReLU -> AdaptiveAvgPool2d -> classifier
            # We need features + ReLU + AdaptiveAvgPool2d to get [B, C] features
            self.net_1 = nn.Sequential(
                model.net.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.net_2 = model.net.classifier
            # Determine feature dimension based on DenseNet variant
            if cnn_name == 'densenet121':
                self.feature_dim = 1024
            elif cnn_name == 'densenet161':
                self.feature_dim = 2208
            elif cnn_name == 'densenet169':
                self.feature_dim = 1664
            elif cnn_name == 'densenet201':
                self.feature_dim = 1920
            else:
                self.feature_dim = 1024  # default fallback
        elif self.use_vit:
            # For ViT, we need to handle the feature extraction differently
            # Store the full ViT model and extract features in forward pass
            self.net_1 = model.net  # Store the full ViT model
            self.net_2 = model.net.heads.head
            # ViT-B/16 outputs 768 features (this is the only ViT variant currently supported)
            # For any ViT variant, use dynamic detection as fallback
            try:
                # Get the input features of the classifier head
                self.feature_dim = model.net.heads.head.in_features
            except:
                self.feature_dim = 768  # default for ViT-B/16
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2
            # Determine feature dimension for other CNN architectures
            # For VGG/AlexNet, we need to get the actual flattened feature dimension
            # The classifier's first layer expects the flattened features from net_1
            if cnn_name in ['alexnet', 'vgg11', 'vgg16', 'vgg19']:
                # Get the actual feature dimension from the classifier's first layer
                # This is the input dimension that the classifier expects after flattening net_1 output
                try:
                    first_linear = list(self.net_2.children())[0]
                    if hasattr(first_linear, 'in_features'):
                        self.feature_dim = first_linear.in_features
                    else:
                        # Fallback: VGG16 typically outputs 25088 (512*7*7) when flattened
                        # AlexNet outputs 9216, VGG11/VGG19 similar to VGG16
                        if cnn_name == 'vgg16':
                            self.feature_dim = 25088
                        elif cnn_name == 'alexnet':
                            self.feature_dim = 9216
                        else:
                            self.feature_dim = 25088  # VGG11/VGG19
                except:
                    # Ultimate fallback based on architecture
                    if cnn_name == 'vgg16':
                        self.feature_dim = 25088
                    elif cnn_name == 'alexnet':
                        self.feature_dim = 9216
                    else:
                        self.feature_dim = 25088  # VGG11/VGG19
            else:
                # Fallback: try to detect from model
                try:
                    # For models with classifier, get the input features
                    if hasattr(model.net_2, 'in_features'):
                        self.feature_dim = model.net_2.in_features
                    else:
                        # Try to get from the first layer of classifier
                        first_layer = list(model.net_2.children())[0]
                        if hasattr(first_layer, 'in_features'):
                            self.feature_dim = first_layer.in_features
                        else:
                            # Try to get from the last layer
                            last_layer = list(model.net_2.children())[-1]
                            if hasattr(last_layer, 'in_features'):
                                self.feature_dim = last_layer.in_features
                            else:
                                self.feature_dim = 512  # ultimate fallback
                except:
                    self.feature_dim = 512  # ultimate fallback
        if self.num_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            vertices = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                        [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                        [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                        [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        elif self.num_views == 12:
            phi = np.sqrt(3)
            vertices = [[1, 0, phi/3], [phi/2, -1/2, phi/3], [1/2,-phi/2,phi/3],
                        [0, -1, phi/3], [-1/2, -phi/2, phi/3],[-phi/2, -1/2, phi/3],
                        [-1, 0, phi/3], [-phi/2, 1/2, phi/3], [-1/2, phi/2, phi/3],
                        [0, 1 , phi/3], [1/2, phi / 2, phi/3], [phi / 2, 1/2, phi/3]]
        else:
            # Use Fibonacci sphere sampling for arbitrary number of views
            # This generates evenly-distributed points on a unit sphere
            vertices = []
            golden_ratio = (1 + np.sqrt(5)) / 2
            for i in range(self.num_views):
                theta = 2 * np.pi * i / golden_ratio  # azimuthal angle
                phi_angle = np.arccos(1 - 2 * (i + 0.5) / self.num_views)  # polar angle
                x = np.sin(phi_angle) * np.cos(theta)
                y = np.sin(phi_angle) * np.sin(theta)
                z = np.cos(phi_angle)
                vertices.append([x, y, z])
            print(f"Using Fibonacci sphere sampling for {self.num_views} views")
        
        # Normalize vertices to unit vectors
        vertices = np.array(vertices, dtype=np.float32)
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Register vertices as a buffer so it automatically moves with the model to the correct device
        # This avoids NVML initialization errors from calling .cuda() during __init__
        self.register_buffer('vertices', torch.tensor(vertices, dtype=torch.float))

        # Compute number of nodes at each level
        # level_nodes[i] is the number of nodes at level i (before any selection)
        # Node selection only happens BETWEEN levels (to pass to next level)
        # So for num_levels=L, we have L levels and (L-1) selections
        # When num_levels <= 3: each selection = 1/2 of previous
        # When num_levels > 3: each selection = 3/4 of previous (gentler reduction for deeper networks)
        self.level_nodes = [self.num_views]  # Level 0 starts with all views
        current_nodes = self.num_views
        for level in range(1, self.num_levels):  # Selections happen between levels
            if self.num_levels <= 3:
                # Standard reduction: 1/2 at each selection
                current_nodes = max(1, current_nodes // 2)
            else:
                # Gentler reduction for deeper networks: 3/4 at each selection
                current_nodes = max(1, current_nodes * 3 // 4)
            self.level_nodes.append(current_nodes)
        
        print(f"View-GCN with {self.num_levels} levels, nodes per level: {self.level_nodes}")
        print(f"Using graph network: {self.graph_net.upper()}")

        # Select graph network class based on graph_net parameter
        if self.graph_net == 'gat':
            GraphNetClass = GlobalGAT
        elif self.graph_net == 'gcn':
            GraphNetClass = GlobalGCN
        else:
            raise ValueError(f"Unsupported graph_net: {self.graph_net}. Must be 'gat' or 'gcn'.")

        # Global graph layers (dense connectivity over all views)
        # Use configurable number of attention heads for each layer
        self.global_gats = nn.ModuleList([
            GraphNetClass(in_channels=self.feature_dim, out_channels=self.feature_dim, 
                         heads=self.n_attn_heads[i], edge_dim=edge_dim)
            for i in range(self.num_levels)
        ])
        
        # Attention pooling heads for different levels
        self.attention_pools = nn.ModuleList([
            AttentionPooling(in_channels=self.feature_dim, hidden_dim=256, 
                           num_heads=self.n_attn_heads[i])
            for i in range(self.num_levels)
        ])
        
        # Saliency scorers for node selection
        self.scorers = nn.ModuleList([
            SaliencyScorer(in_channels=self.feature_dim, hidden_dim=128, att_lambda=att_lambda)
            for _ in range(self.num_levels)
        ])
        
        # Top-k selectors for hierarchical selection
        self.topk_selectors = nn.ModuleList([
            TopKSelector(temperature=1.0, straight_through=diff_topk)
            for _ in range(self.num_levels)
        ])

        # Final classifier - input dimension is feature_dim * num_levels (pooled features from each level)
        self.cls = nn.Sequential(
            nn.Linear(self.feature_dim * self.num_levels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.nclasses)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def update_temperatures(self, step: int, total_steps: int,
                            tau_start: float = 2.0, tau_end: float = 0.35,
                            logit_tau_start: float = 1.2, logit_tau_end: float = 0.9):
        # cosine anneal from warm to sharp
        r = step / max(1, total_steps)
        tau  = tau_end  + 0.5 * (tau_start  - tau_end)  * (1.0 + math.cos(math.pi * r))
        ltau = logit_tau_end + 0.5 * (logit_tau_start - logit_tau_end) * (1.0 + math.cos(math.pi * r))
        for t in self.topk_selectors:
            t.set_temperature(tau=tau, logit_tau=ltau)


    
    def forward(self, x):
        BV = x.size(0)
        V  = self.num_views
        assert BV % V == 0, f"Input batch must be multiples of num_views={V}"
        B  = BV // V

        # Encode views -> [B, V, C]
        if hasattr(self, "_resnet_feats"):
            f = self._resnet_feats(x)         # [B*V, C]
        else:
            # Fallback if your class names differ
            f = self.net_1(x)                 # [B*V, C,1,1] -> pool outside if needed
            f = torch.flatten(f, 1)

        F = f.view(B, V, -1)                   # [B, V, C]
        C = F.size(-1)

        # Canonical camera vertices replicated per batch
        V_xyz = self.vertices.unsqueeze(0).repeat(B, 1, 1).to(F.device)   # [B, V, 3]

        use_diff = getattr(self, "diff_topk", False)
        
        # Storage for outputs from each level
        pooled_features = []
        all_saliency_logits = []
        all_saliency_probs = []
        all_topk_indices = []
        all_topk_weights = []
        all_attn_weights = []
        all_soft_masks = []

        # Current features and vertices for hierarchical processing
        F_current = F
        V_current = V_xyz

        # Process each level
        for level in range(self.num_levels):
            # Apply GAT/GCN
            F_out, A_out = self.global_gats[level](F_current, V_current, return_attention=True)
            
            # Score nodes
            logits_out, probs_out = self.scorers[level](F_out, A_out)
            
            # Check if this is the last level
            is_last_level = (level == self.num_levels - 1)
            
            if is_last_level:
                # Last level: pool all nodes, no selection needed
                z, W = self.attention_pools[level](F_out)
                # z = F_out.mean(dim=1) 
                # W = None
                
                # Store outputs (use F_out for features, no selection)
                pooled_features.append(z)
                all_saliency_logits.append(logits_out)
                all_saliency_probs.append(probs_out)
                all_topk_indices.append(torch.arange(F_out.size(1), device=F_out.device).unsqueeze(0).expand(F_out.size(0), -1))
                all_topk_weights.append(torch.ones(F_out.size(0), F_out.size(1), device=F_out.device) / F_out.size(1))
                all_attn_weights.append(W)
                all_soft_masks.append(torch.ones(F_out.size(0), F_out.size(1), device=F_out.device))
            else:
                # Not last level: select top-k nodes for next level
                k = self.level_nodes[level + 1]  # Number of nodes for next level
                
                # Select top-k nodes
                F_sel, idx, w, mask, M = self.topk_selectors[level](F_out, logits_out, k=k)
                
                # Update vertices based on selection
                if use_diff:
                    # V_sel = torch.bmm(M, V_current)
                    idx_long = idx.to(device=V_current.device, dtype=torch.long)
                    V_sel = V_current.gather(1, idx_long.unsqueeze(-1).expand(-1, -1, 3))
                else:
                    V_sel = V_current.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
                
                # Pool features for this level (pool selected features)
                z, W = self.attention_pools[level](F_sel)
                
                # Store outputs
                pooled_features.append(z)
                all_saliency_logits.append(logits_out)
                all_saliency_probs.append(probs_out)
                all_topk_indices.append(idx)
                all_topk_weights.append(w)
                all_attn_weights.append(W)
                all_soft_masks.append(mask)
                
                # Update current features and vertices for next level
                F_current = F_sel
                V_current = V_sel

        # Classifier - concatenate pooled features from all levels
        logits = self.cls(torch.cat(pooled_features, dim=-1))         # [B, nclasses]

        # Return optional info for logging/regularization
        return logits, {
            'saliency_logits': tuple(all_saliency_logits),
            'saliency_probs':  tuple(all_saliency_probs),
            'topk_indices':    tuple(all_topk_indices),
            'topk_weights':    tuple(all_topk_weights),
            'attn_weights':    tuple(all_attn_weights),
            'soft_masks':      tuple(all_soft_masks),
        }

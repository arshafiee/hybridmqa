from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv
from torchvision.models import ResNet, ResNet50_Weights


class ConvBlock(nn.Module):
    """
    A two-layer convolutional block with optional residual connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (Optional[int]): Number of hidden channels; defaults to `out_channels` if not provided.
        residual (bool, optional): Whether to apply a residual connection from input to output. Defaults to False.
        kernel_size (int, optional): Size of the convolution kernels. Defaults to 3.
        stride (List[int], optional): List of two stride values for the two conv layers. Defaults to [1, 1].
        padding (int, optional): Amount of zero-padding added to both sides of the input. Defaults to 1.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        residual: bool = False,
        kernel_size: int = 3,
        stride: List[int] = [1, 1],
        padding: int = 1
    ) -> None:
        super(ConvBlock, self).__init__()
        self.residual = residual
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size,
                               stride=stride[0], padding=padding)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size,
                               stride=stride[1], padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [N, C_out, H_out, W_out].
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual:
            out += identity
        out = self.act(out)

        return out


class CrsAttTrans(nn.Module):
    """
    A single Transformer block implementing multi-head cross-attention followed by
    a feedforward network with residual connections and layer normalization.

    Args:
        embed_dim (int): Dimensionality of input embeddings.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        hidden_dim (int, optional): Hidden dimension in the feedforward network. Defaults to 128.
        dropout (float, optional): Dropout rate applied after attention and feedforward layers. Defaults to 0.1.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super(CrsAttTrans, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor, att_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the cross-attention transformer block.

        Args:
            x (torch.Tensor): Tensor to be used as the key and value, shape [L, N, C].
            y (torch.Tensor): Tensor to be used as query, shape [L, N, C].
            att_mask (Optional[torch.Tensor]): Optional attention mask of shape [N*num_heads, L, L] or None.

        Returns:
            torch.Tensor: Output tensor of shape [L, N, C] after attention and feedforward fusion.
        """
        # apply multi-head attention
        out, _ = self.attention(y, x, x, attn_mask=att_mask)
        out = self.dropout1(out) + x
        out = self.norm1(out)
        # apply feedforward layer
        out2 = self.feedforward(out)
        out2 = self.dropout2(out2) + out
        out = self.norm2(out2)

        return out


class CrsAttModule(nn.Module):
    """
    A cross-attention module that models geometry-texture interactions using symmetric multi-head attention.
    Optionally supports spatial attention masking.

    This module takes two inputs (e.g., RGB and 3D features), applies cross-attention
    in both directions, and returns a fused representation by concatenating both outputs.

    Args:
        embed_dim (int): Embedding dimension of the input features.
        num_heads (int): Number of attention heads.
        mask_flag (bool): Whether to use binary attention masks to ignore background regions.
    """
    def __init__(self, embed_dim: int, num_heads: int, mask_flag: bool = False):
        super(CrsAttModule, self).__init__()
        self.num_heads = num_heads
        self.mask_flag = mask_flag
        self.crs_attn1 = CrsAttTrans(embed_dim=embed_dim, num_heads=num_heads)
        self.crs_attn2 = CrsAttTrans(embed_dim=embed_dim, num_heads=num_heads)

    def prepare_att_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Prepares a boolean attention mask for multi-head attention and final pooling based on a spatial mask.

        Args:
            mask (torch.Tensor): Binary mask of shape [H*W, N] indicating valid regions.

        Returns:
            torch.Tensor: Attention mask of shape [N * num_heads, H*W, H*W].
                          True entries indicate positions to be ignored during attention.
        """
        mask = mask.unsqueeze(2).permute(1, 0, 2)  # shape (N, L, 1), L = H*W
        attention_mask = torch.mul(torch.ones(mask.size(), dtype=torch.bool, device=mask.device),
                                   mask.permute(0, 2, 1))  # shape (N, L, L)
        attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # shape (N, num_heads, L, L)
        N, _, L, _ = attention_mask.shape
        attention_mask = attention_mask.reshape(N * self.num_heads, L, L)  # shape (N * num_heads, L, L)
        attention_mask = ~attention_mask  # true indicates unattendable indices

        return attention_mask

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the cross-attention module.

        Args:
            x (torch.Tensor): First input feature tensor of shape [N, C, H, W].
            y (torch.Tensor): Second input feature tensor of shape [N, C, H, W].
            mask (torch.Tensor): Binary mask of shape [N, 1, H, W] indicating valid spatial regions.

        Returns:
            torch.Tensor: Fused output tensor of shape [N, 2 * C] (concatenated from both attention directions).
        """
        x_flatten = x.flatten(2).permute(2, 0, 1)  # shape (H*W, N, C)
        y_flatten = y.flatten(2).permute(2, 0, 1)  # shape (H*W, N, C)
        mask = mask.flatten(2).permute(2, 0, 1).squeeze()  # shape (H*W, N)
        attention_mask = self.prepare_att_mask(mask) if self.mask_flag else None  # shape (N * num_heads, H*W, H*W)

        attn_output1 = self.crs_attn1(x_flatten, y_flatten, attention_mask)  # shape (H*W, N, C)
        attn_output2 = self.crs_attn2(y_flatten, x_flatten, attention_mask)  # shape (H*W, N, C)
        attn_output = torch.cat((attn_output1, attn_output2), dim=-1)  # shape (H*W, N, 2C)

        # average masked pooling
        if self.mask_flag:
            mask = mask.unsqueeze(2).expand(-1, -1, attn_output.shape[2])
            attn_output = torch.sum(attn_output * mask, dim=0)
            attn_output = attn_output / torch.sum(mask, dim=0)
        else:
            attn_output = torch.mean(attn_output, dim=0)  # shape (N, 2C)

        return attn_output


class CrsAttResNet(ResNet):
    """
    CrsAttResNet contains the Image (resnet-based) and 3D Feature Encoders plus the Cross-Attention modules.

    Args:
        feat_in_dim (int): Input channel dimension of feature maps.
        feat_hidden_dim (int): Hidden channel dimension for the feature pathway.
        out_dim (int): Output embedding dimension.
        weights (ResNet50_Weights): pre-trained weights to load.
        mask_attn (bool): Whether to use binary attention masks for selective focus. Defaults to False.
        *args, **kwargs: Additional arguments passed to `ResNet` base class (e.g., `block`, `layers`).
    """
    def __init__(
        self,
        feat_in_dim: int,
        feat_hidden_dim: int,
        out_dim: int,
        weights: ResNet50_Weights,
        mask_attn: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # init weights with pretrained resnet
        if weights is not None:
            self.load_state_dict(weights.get_state_dict(progress=True), strict=False)

        resnet_hidden_dims = [64, 256, 512, 1024, 2048]
        self.feat_convs = nn.ModuleList()
        self.rgb_heads = nn.ModuleList()
        self.feat_heads = nn.ModuleList()
        self.cross_atts = nn.ModuleList()

        # feature tower and cross attention heads
        # att dim is half of the output dim because of the symmetric cross attention
        att_dim = out_dim // 2
        for layer, stride in enumerate([1, 1, 2, 2, 2]):
            in_dim = feat_in_dim if layer == 0 else feat_hidden_dim
            self.feat_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, feat_hidden_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(feat_hidden_dim),
                    nn.ReLU(inplace=True)
                )
            )
            self.feat_heads.append(
                nn.Conv2d(feat_hidden_dim, att_dim, kernel_size=1, stride=1)
            )
            self.rgb_heads.append(
                nn.Conv2d(resnet_hidden_dims[layer], att_dim, kernel_size=1, stride=1)
            )
            self.cross_atts.append(
                CrsAttModule(att_dim, num_heads=4, mask_flag=mask_attn)
            )

    def forward(self, x: torch.Tensor, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CrsAttResNet module.

        Args:
            x (torch.Tensor): Input RGB image tensor.
            feat (torch.Tensor): 3D Feature Projections
            mask (torch.Tensor): Attention mask indicating non-background regions

        Returns:
            torch.Tensor: Concatenated outputs of cross attention modules.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_layer0 = self.maxpool(x)
        x_layer1 = self.layer1(x_layer0)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x_layers = [x_layer0, x_layer1, x_layer2, x_layer3, x_layer4]

        out = None  # placeholder for the output
        for idx, (x_layer, feat_conv, feat_head, rgb_head, cross_att) in enumerate(zip(
                x_layers, self.feat_convs, self.feat_heads, self.rgb_heads, self.cross_atts)):
            mask = F.interpolate(mask.float(), scale_factor=0.5, mode='bilinear') > 0.0 if idx > 1 else mask
            feat = feat_conv(feat)
            feat_layer_head = feat_head(feat)
            x_layer_head = rgb_head(x_layer)
            out_layer = cross_att(x_layer_head, feat_layer_head, mask)  # shape (N, 2C)
            out = torch.cat((out, out_layer), dim=-1) if idx > 0 else out_layer

        return out


class GCN(nn.Module):
    """
    A multi-layer Graph Convolutional Network (GCNConv-based).

    Args:
        in_channels (int): Number of input node feature dimensions.
        hidden_channels (int): Number of hidden node feature dimensions.
        out_channels (int): Number of output node feature dimensions.
        layers (int): Number of GraphConv layers in total.
        gnn_act (str): Activation function to use. Options: 'relu', 'lrelu', 'elu'.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layers: int,
        gnn_act: str
    ) -> None:
        super(GCN, self).__init__()
        self.layers = layers
        if gnn_act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif gnn_act == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        elif gnn_act == 'elu':
            self.act = nn.ELU()
        else:
            raise ValueError(f'{gnn_act} activation function is not supported!')
        for layer in range(layers):
            if layer == 0:
                setattr(self, f'conv{layer+1}', GCNConv(in_channels, hidden_channels))
            elif layer == layers - 1:
                setattr(self, f'conv{layer+1}', GCNConv(hidden_channels, out_channels))
            else:
                setattr(self, f'conv{layer+1}', GCNConv(hidden_channels, hidden_channels))

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the stacked GCNConv layers.

        Args:
            data (Data): A PyTorch Geometric `Data` object with node features `x`, `edge_index`, and `edge_weight`.

        Returns:
            torch.Tensor: Updated node features after GNN processing.
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.conv1(x, edge_index, edge_weight)
        for layer in range(self.layers - 1):
            x = self.act(x)
            x = getattr(self, f'conv{layer+2}')(x, edge_index, edge_weight)

        data.x = x

        return data


class GAT(nn.Module):
    """
    A multi-layer Graph Convolutional Network (GATConv-based).

    Args:
        in_channels (int): Number of input node feature dimensions.
        hidden_channels (int): Number of hidden node feature dimensions.
        out_channels (int): Number of output node feature dimensions.
        layers (int): Number of GraphConv layers in total.
        gnn_act (str): Activation function to use. Options: 'relu', 'lrelu', 'elu'.
        heads (int): Number of attention heads for GATConv. Defaults to 4.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layers: int,
        gnn_act: str,
        heads: int = 4
    ) -> None:
        super(GAT, self).__init__()
        self.layers = layers
        if gnn_act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif gnn_act == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        elif gnn_act == 'elu':
            self.act = nn.ELU()
        else:
            raise ValueError(f'{gnn_act} activation function is not supported!')
        for layer in range(layers):
            if layer == 0:
                setattr(self, f'conv{layer+1}', GATConv(in_channels, hidden_channels, heads=heads))
            elif layer == layers - 1:
                setattr(self, f'conv{layer+1}', GATConv(hidden_channels * heads, out_channels, heads=1))
            else:
                setattr(self, f'conv{layer+1}', GATConv(hidden_channels * heads, hidden_channels, heads=heads))

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the stacked GATConv layers.

        Args:
            data (Data): A PyTorch Geometric `Data` object with node features `x`, `edge_index`, and `edge_weight`.

        Returns:
            torch.Tensor: Updated node features after GNN processing.
        """
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        for layer in range(self.layers - 1):
            x = self.act(x)
            x = getattr(self, f'conv{layer+2}')(x, edge_index)

        # update graph data
        data.x = x

        return data


class GraphSAGE(nn.Module):
    """
    A multi-layer Graph Convolutional Network (SAGEConv-based).

    Args:
        in_channels (int): Number of input node feature dimensions.
        hidden_channels (int): Number of hidden node feature dimensions.
        out_channels (int): Number of output node feature dimensions.
        layers (int): Number of GraphConv layers in total.
        gnn_act (str): Activation function to use. Options: 'relu', 'lrelu', 'elu'.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layers: int,
        gnn_act: str
    ) -> None:
        super(GraphSAGE, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        if gnn_act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif gnn_act == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        elif gnn_act == 'elu':
            self.act = nn.ELU()
        else:
            raise ValueError(f'{gnn_act} activation function is not supported!')
        for layer in range(layers):
            if layer == 0:
                self.convs.append(SAGEConv(in_channels, hidden_channels, project=True))
            elif layer == layers - 1:
                self.convs.append(SAGEConv(hidden_channels, out_channels, project=True))
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, project=True))

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the stacked SAGEConv layers.

        Args:
            data (Data): A PyTorch Geometric `Data` object with node features `x`, `edge_index`, and `edge_weight`.

        Returns:
            torch.Tensor: Updated node features after GNN processing.
        """
        x, edge_index = data.x, data.edge_index

        for layer in range(self.layers):
            x = self.convs[layer](x, edge_index)
            if layer < self.layers - 1:  # Apply activation and dropout except for the last layer
                x = self.act(x)

        data.x = x

        return data


class Graph(nn.Module):
    """
    A multi-layer Graph Convolutional Network (GraphConv-based).

    Args:
        in_channels (int): Number of input node feature dimensions.
        hidden_channels (int): Number of hidden node feature dimensions.
        out_channels (int): Number of output node feature dimensions.
        layers (int): Number of GraphConv layers in total.
        gnn_act (str): Activation function to use. Options: 'relu', 'lrelu', 'elu'.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layers: int,
        gnn_act: str
    ) -> None:
        super(Graph, self).__init__()
        self.layers = layers
        self.convs = torch.nn.ModuleList()
        if gnn_act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif gnn_act == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        elif gnn_act == 'elu':
            self.act = nn.ELU()
        else:
            raise ValueError(f'{gnn_act} activation function is not supported!')
        for layer in range(layers):
            if layer == 0:
                self.convs.append(GraphConv(in_channels, hidden_channels))
            elif layer == layers - 1:
                self.convs.append(GraphConv(hidden_channels, out_channels))
            else:
                self.convs.append(GraphConv(hidden_channels, hidden_channels))

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the stacked GraphConv layers.

        Args:
            data (Data): A PyTorch Geometric `Data` object with node features `x`, `edge_index`, and `edge_weight`.

        Returns:
            torch.Tensor: Updated node features after GNN processing.
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        for layer in range(self.layers):
            x = self.convs[layer](x, edge_index, edge_weight)
            if layer < self.layers - 1:  # Apply activation and dropout except for the last layer
                x = self.act(x)

        data.x = x

        return data

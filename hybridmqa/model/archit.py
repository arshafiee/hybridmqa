from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchvision import models
from torchvision.models import ResNet50_Weights

from hybridmqa.model.backbone import (GAT, GCN, ConvBlock, CrsAttResNet, Graph,
                                      GraphSAGE)
from hybridmqa.utils import (apply_hflip_and_vflip, construct_batch_graph,
                             patchify_and_generate_mask, pool_patch_features,
                             render_projections)


class BaseEncoder(nn.Module):
    """
    A simple convolutional encoder block that extracts feature maps from 2D texture, normal, and vertex maps.

    Args:
        hidden_channels (int): Number of channels in the hidden layer of the convolutional block.
        out_channels (int): Number of output channels after encoding.
        input_maps (str, optional): Specifies which input maps to use.
                Options: 'norm_ver', 'norm', 'tex_norm', 'tex_ver', 'all'. Defaults to 'all'.
    """
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        input_maps: str = 'all',
    ) -> None:
        super(BaseEncoder, self).__init__()
        input_ch_dict = {'norm_ver': 6, 'norm': 3, 'tex_norm': 6, 'tex_ver': 6, 'all': 9}
        self.input_maps = input_maps
        if self.input_maps not in input_ch_dict.keys():
            raise ValueError(f'Input map selection ({self.input_maps}) not supported!')

        self.enc_in_channels = input_ch_dict[self.input_maps]
        self.conv1 = ConvBlock(in_channels=self.enc_in_channels, out_channels=out_channels,
                               hidden_channels=hidden_channels, residual=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C_in, H, W].

        Returns:
            torch.Tensor: Encoded tensor of shape [B, C_out, H, W].
        """
        if self.input_maps == 'norm_ver':
            x_in = x[:, 3:, ...]
        elif self.input_maps == 'norm':
            x_in = x[:, 3:6, ...]
        elif self.input_maps == 'tex_norm':
            x_in = x[:, :6, ...]
        elif self.input_maps == 'tex_ver':
            x_in = torch.cat((x[:, :3, ...], x[:, 6:, ...]), dim=1)
        else:  # all
            x_in = x

        out = self.conv1(x_in)

        return out


class GNN(nn.Module):
    """
    A wrapper class for building various Graph Neural Network (GNN) architectures.

    Supported GNN types include:
        - GCNConv
        - GATConv
        - SAGEConv
        - GraphConv

    Args:
        in_channels (int): Number of input node feature dimensions.
        hidden_channels (int): Hidden feature dimension in the GNN layers.
        out_channels (int): Number of output feature dimensions.
        gnn_arch (str, optional): Architecture specification string, e.g., 'gcn_2', 'gat_3_4'.
                                    Format: <type>_<layers>[_<heads>] where type ∈ {gcn, gat, sage, graph}.
                                    Defaults to 'graph_2'.
        gnn_act (str, optional): Activation function to use. Options: 'relu', 'lrelu', 'elu'. Defaults to 'relu'.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        gnn_arch: str = 'graph_2',
        gnn_act: str = 'relu'
    ) -> None:
        super(GNN, self).__init__()

        gnn_type, layers, *rest = gnn_arch.split('_')
        layers = int(layers)
        heads = int(rest[0]) if gnn_type == 'gat' else None

        if gnn_type == 'gcn':
            self.gnn = GCN(in_channels, hidden_channels, out_channels, layers, gnn_act=gnn_act)
        elif gnn_type == 'gat':
            self.gnn = GAT(in_channels, hidden_channels, out_channels, layers, heads, gnn_act=gnn_act)
        elif gnn_type == 'sage':
            self.gnn = GraphSAGE(in_channels, hidden_channels, out_channels, layers, gnn_act=gnn_act)
        elif gnn_type == 'graph':
            self.gnn = Graph(in_channels, hidden_channels, out_channels, layers, gnn_act=gnn_act)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

    def forward(self, data: Data) -> torch.Tensor:
        """
        Performs a forward pass on the input graph data.

        Args:
            data (Data): A PyTorch Geometric `Data` object containing graph inputs.

        Returns:
            torch.Tensor: Output node embeddings after processing through the selected GNN.
        """
        return self.gnn(data)


class QualityEncoder(nn.Module):
    """
    The QualityEncoder encodes the two sets of renderings and models the geometry-texture interactions
    using cross-attention and obtains the mesh quality representation.

    Args:
        in_channels (int): Number of input feature channels.
        hidden_channels (int): Hidden feature dimension for convolutional layers.
        out_channels (int): Output feature dimension from CrsAttResNet.
        mask_attn (bool, optional): Whether to use attention masking during cross-attention. Defaults to False.
        num_proj (int, optional): Number of projections per mesh used in training. Defaults to 6.
        patch_size (int, optional): Patch size used during patchification. Defaults to 64.
        nonempty_ratio (float, optional): Minimum ratio of non-background pixels in a patch. Defaults to 0.1.
        stride_ratio (float, optional): Stride-to-patch size ratio for patchification. Defaults to 1.0.
        flip_aug (bool, optional): Whether to use flip augmentation during training. Defaults to False.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        mask_attn: bool = False,
        num_proj: int = 6,
        patch_size: int = 64,
        nonempty_ratio: float = 0.1,
        stride_ratio: float = 1.0,
        flip_aug: bool = False
    ) -> None:
        super(QualityEncoder, self).__init__()
        self.num_proj = num_proj
        self.patch_size = patch_size
        self.nonempty_ratio = nonempty_ratio
        self.stride_ratio = stride_ratio
        self.flip_aug = flip_aug
        self.crsattresnet = CrsAttResNet(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            weights=ResNet50_Weights.verify(ResNet50_Weights.IMAGENET1K_V2),
            feat_in_dim=in_channels,
            feat_hidden_dim=hidden_channels,
            out_dim=out_channels,
            mask_attn=mask_attn
        )

        # graph feature processing
        self.feat_module = ConvBlock(in_channels, in_channels, residual=True)

    def forward(self, projections: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the Quality Encoder model.

        Args:
            projections (Tuple[torch.Tensor, torch.Tensor]):
                A tuple of tensors: (RGB projections, feature projections).
                - RGB: shape [N, P, H, W, C1]
                - Feature: shape [N, P, H', W', C2]

        Returns:
            torch.Tensor: Feature tensor of shape [N, C] representing the mesh quality representation.
        """
        N = projections[0].shape[0]
        eff_num_proj = self.num_proj if self.training else 6
        # patchify and generate masks
        patchify_out = patchify_and_generate_mask(
            projections,
            patch_size=self.patch_size,
            background_intensity=0.0,
            nonempty_ratio=self.nonempty_ratio,
            stride_ratio=self.stride_ratio
        )
        projections, nonempty_patches, bg_mask = patchify_out
        # flip augmentations
        if self.flip_aug and self.training:
            projections, bg_mask = apply_hflip_and_vflip(projections, bg_mask)

        # quality encoding
        rgb, feat = projections[0], projections[1]
        rgb = rgb.permute(0, 3, 1, 2)
        feat = feat.permute(0, 3, 1, 2) if feat is not None else None
        mask = bg_mask.permute(0, 3, 1, 2)
        # downsample mask twice to match "feat" dimensionality
        mask = F.interpolate(mask.float(), scale_factor=0.5, mode='bilinear') > 0.0
        mask = F.interpolate(mask.float(), scale_factor=0.5, mode='bilinear') > 0.0
        x = rgb[:, :3, ...]  # Assuming x is now [num_nonempty_patches, 3, H, W]
        reg_feat = self.crsattresnet(x, feat, mask)
        # add feat processing
        graph_feat = self.feat_module(feat)  # shape [num_nonempty_patches, C2, H, W]
        # pooling over the mask
        graph_feat = (graph_feat * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3])
        # shape [num_nonempty_patches, C2]

        # patch pooling
        # reg_feat of shape [N*P, regressor_out_dim], graph_feat of shape [N*P, graph_out_dim]
        reg_feat, graph_feat = pool_patch_features(
            N=N,
            num_proj=eff_num_proj,
            reg_feat=reg_feat,
            graph_feat=graph_feat,
            nonempty_patches=nonempty_patches
        )
        # concat all features
        x_feat_out = torch.cat((graph_feat, reg_feat), dim=-1)

        return x_feat_out


class HybridMQA(nn.Module):
    """
    HybridMQA is the main model class combining a CNN encoder, a GNN,
    and a regression head for quality representation.

    Components:
        - Base Encoder: extracts feature maps from 2D maps.
        - GNN: processes graph-structured mesh representations.
        - Quality Encoder: models geometry-texture interactions to obtain mesh quality representation.
        - Final linear head: outputs a scalar quality score.
    Args:
        base_enc (nn.Module): Feature extractor for 2D maps.
        gnn (nn.Module): Graph neural network for 3D mesh representation.
        quality_enc (nn.Module): module for modeling geometry-texture interactions.
        fc_input_dim (int): Input dimension for final regression head.
        img_size (int, optional): Image size for rendering. Defaults to 512.
        num_proj (int, optional): Number of projections per mesh. Defaults to 6.
        angle_aug (bool, optional): Enable angle augmentation during training. Defaults to False.
        lighting (str, optional): Lighting scheme, either 'directional' or 'ambient'. Defaults to 'directional'.
        vcmesh (bool, optional): If True, assumes the mesh is a vertex-color mesh (not textured). Defaults to False.
    """
    def __init__(
        self,
        base_enc: nn.Module,
        gnn: nn.Module,
        quality_enc: nn.Module,
        fc_input_dim: int,
        img_size: int = 512,
        num_proj: int = 6,
        angle_aug: bool = False,
        lighting: str = 'directional',
        vcmesh: bool = False,
    ) -> None:
        super(HybridMQA, self).__init__()
        self.base_enc = base_enc
        self.gnn = gnn
        self.quality_enc = quality_enc
        self.img_size = img_size
        self.num_proj = num_proj
        self.angle_aug = angle_aug
        self.lighting = lighting
        self.vcmesh = vcmesh

        # register regression and classification heads
        self.reg_head = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, input: Dict[str, Any]) -> torch.Tensor:
        """
        Performs a forward pass through the HybridMQA model.

        Args:
            input (Dict[str, Any]): A dictionary containing:
                - 'input_concat' (torch.Tensor): Concatenated input maps.
                - 'mesh_data' (Meshes): PyTorch3D mesh objects.

        Returns:
            torch.Tensor: Predicted scalar quality scores for each input pair.
        """
        x_cat, x_mesh = input['input_concat'], input['mesh_data']
        # Forward pass
        enc_out = self.base_enc(x_cat) if not self.vcmesh else None  # shape [batch_size, out_channels, 256, 256]
        # construct batch graph
        graph = construct_batch_graph(enc_out, x_mesh, self.vcmesh)
        # pass graph to GNN
        graph_out = self.gnn(graph)
        # render rgb and feature images, shape [batch_size, num_projections, H, W, feature_dim+4]
        projections = render_projections(
            batch_graph=graph_out,
            batch_mesh_data=x_mesh,
            num_proj=self.num_proj,
            img_size=self.img_size,
            training=self.training,
            angle_aug=self.angle_aug,
            lighting=self.lighting
        )
        # run quality encoder
        x_feat_out = self.quality_enc(projections)
        # prepare final output features
        x_feat_out_ref, x_feat_out_dist = x_feat_out[::2, ...], x_feat_out[1::2, ...]
        feat_out = torch.abs(x_feat_out_ref - x_feat_out_dist)
        # make predictions
        pred_quality = self.reg_head(feat_out).squeeze()

        return pred_quality

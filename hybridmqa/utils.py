from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch3d.renderer import (AmbientLights, BlendParams, DirectionalLights,
                                FoVPerspectiveCameras, HardPhongShader,
                                Materials, MeshRasterizer, MeshRenderer,
                                RasterizationSettings, SoftPhongShader,
                                TexturesVertex, look_at_view_transform)
from pytorch3d.structures import Meshes
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_undirected
from torchvision.transforms.functional import hflip, vflip


def calculate_edge_weights(vertices: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Calculate normalized edge weights based on the inverse Euclidean distance
    between connected vertex pairs in a mesh graph.

    Args:
        vertices (torch.Tensor): Tensor of shape [num_vertices, num_features] representing vertex coordinates.
        edge_index (torch.Tensor): LongTensor of shape [2, num_edges] representing graph connectivity.

    Returns:
        torch.Tensor: A 1D tensor of shape [num_edges] containing normalized edge weights.
    """
    # Compute Euclidean distances between pairs of nodes (vertices)
    node_pairs = vertices[edge_index]  # Shape: [2, num_edges, num_features]
    distances = torch.norm(node_pairs[0] - node_pairs[1], dim=1)  # Shape: [num_edges]

    weights = 1.0 / (distances + 1e-6)  # Use reciprocal of distances as edge weights

    # Normalize the weights for each node to sum to 1
    # First, get the sum of weights for each node
    sum_weights = torch.zeros(vertices.shape[0], device=vertices.device)
    sum_weights.index_add_(0, edge_index[0], weights)
    # Normalize the weights
    weights /= sum_weights[edge_index[0]]

    return weights


def construct_mesh_graph(feature_map: torch.Tensor, mesh_data: Meshes, vcmesh: bool) -> Data:
    """
    Constructs a graph from a 3D mesh by defining vertex features and edge weights.

    Args:
        feature_map (torch.Tensor): Feature map from which vertex features are sampled.
            Should be of shape [C, H, W] for non-vcmesh mode.
        mesh_data: Mesh object containing vertices, faces, and textures.
        vcmesh (bool): If True, use vertex colors and normals directly. If False,
            sample features from the input feature_map. Should be used for vertex-color meshes.

    Returns:
        Data: A PyTorch Geometric `Data` object representing the mesh graph, with
            attributes `x` (vertex features), `edge_index`, and `edge_weight`.
    """
    device = mesh_data.device
    verts, face_verts = mesh_data.get_mesh_verts_faces(0)
    if not vcmesh:
        # Initialize feature list and adjacency matrix
        uvs = mesh_data.textures.verts_uvs_list()[0]
        face_uvs = mesh_data.textures.faces_uvs_list()[0]

        vertex_features = torch.zeros((verts.shape[0], feature_map.shape[0]), device=device)
        vertex_visit_count = torch.zeros(verts.shape[0], device=device)

        # Normalized UV coordinates for differentiable sampling
        normalized_uvs = 2.0 * uvs - 1.0
        # grid_sample requires top-left corner to be (-1, -1) and bottom-right corner to be (1, 1)
        # therefore flip v coordinate
        normalized_uvs[:, 1] = -normalized_uvs[:, 1]

        # Flatten the face_uvs to get a long list of UV indices for all vertices in all faces
        flat_uv_indices = face_uvs.view(-1)
        # Get the corresponding UV coordinates and normalize
        flat_uvs = normalized_uvs[flat_uv_indices]
        # Reshape to [1, N, 1, 2] where N is total number of vertices across all faces
        grid = flat_uvs.view(1, -1, 1, 2)

        # Perform grid sampling for all vertices
        sampled_features = F.grid_sample(feature_map.unsqueeze(0), grid, align_corners=True).squeeze()
        sampled_features = sampled_features.permute(1, 0)  # Shape: [N, feature_dim]
        # Vertex features aggregation
        flat_face_verts = face_verts.view(-1)  # Flatten face_verts
        vertex_features.index_add_(0, flat_face_verts, sampled_features)  # Aggregate features
        vertex_visit_count.index_add_(0, flat_face_verts, torch.ones_like(flat_face_verts, dtype=torch.float))

        # Average the features based on the visit count
        nonzero_counts = vertex_visit_count > 0
        vertex_features[nonzero_counts] /= vertex_visit_count[nonzero_counts].unsqueeze(1)
    else:
        verts_colors = mesh_data.textures.verts_features_list()[0]
        verts_normals = mesh_data.verts_normals_list()[0]
        vertex_features = torch.cat((verts_colors, verts_normals, verts), dim=-1)

    edges = torch.cat([face_verts[:, [0, 1]], face_verts[:, [1, 2]], face_verts[:, [2, 0]]], dim=0).t().contiguous()

    # Remove duplicates and make the graph undirected
    edge_index = to_undirected(edges)

    # edge weights
    edge_weights = calculate_edge_weights(verts, edge_index)

    # create a graph Data object in pytorch geometric
    graph = Data(x=vertex_features, edge_index=edge_index, edge_weight=edge_weights)

    return graph


def construct_batch_graph(
    feature_maps: Union[torch.Tensor, torch.Tensor],
    batch_mesh_data: Meshes,
    vcmesh: bool
) -> Batch:
    """
    Constructs a batched PyTorch Geometric graph from a batch of 3D mesh data
    and their associated feature maps.

    Args:
        feature_maps (Union[torch.Tensor, torch.Tensor]): Tensor of shape [B, C, H, W] containing
            feature maps for each mesh in the batch. If `vcmesh` is True, this can be a dummy
            tensor as vertex features are used instead.
        batch_mesh_data (Meshes): A batch of PyTorch3D `Meshes` objects.
        vcmesh (bool): Flag indicating whether to use vertex colors and normals directly
            instead of sampling from `feature_maps`. Should be used for vertex-color meshes.

    Returns:
        Batch: A PyTorch Geometric `Batch` object combining individual mesh graphs.
    """
    graphs = []
    feature_maps = torch.zeros(len(batch_mesh_data)) if vcmesh else feature_maps
    for feature_map, mesh_data in zip(feature_maps, batch_mesh_data):
        graph = construct_mesh_graph(feature_map, mesh_data, vcmesh)
        graphs.append(graph)
    return Batch.from_data_list(graphs)


def render_rgb_projections(
    batch_mesh_data: Meshes,
    img_size: int,
    num_proj: int,
    elev: torch.Tensor,
    azim: torch.Tensor,
    lighting: str,
    dist: float
) -> torch.Tensor:
    """
    Renders RGB projections of a batch of 3D meshes from multiple viewpoints.

    Args:
        batch_mesh_data (Meshes): A batch of PyTorch3D `Meshes` to be rendered.
        img_size (int): Desired image size (H and W) for the rendered output.
        num_proj (int): Number of viewpoints to render per mesh.
        elev (torch.Tensor): Tensor of shape [num_proj] specifying elevation angles.
        azim (torch.Tensor): Tensor of shape [num_proj] specifying azimuth angles.
        lighting (str): Lighting scheme, either 'directional' or 'ambient'.
        dist (float): Distance from camera to object.

    Returns:
        Tensor: A tensor of shape [batch_size, num_proj, H, W, 4] representing rendered RGBA images.
    """
    device = batch_mesh_data.device
    list_images = []

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    if lighting == 'directional':
        light_direction_camera = torch.tensor([[-1.0], [1.0], [-1.0]])  # top-right-out
        light_directions = torch.matmul(R, light_direction_camera).squeeze()
        lights = DirectionalLights(
            ambient_color=((0.3, 0.3, 0.3),),
            diffuse_color=((0.7, 0.7, 0.7),),
            specular_color=((0.0, 0.0, 0.0),),
            direction=light_directions,
            device=device
        )
    elif lighting == 'ambient':
        lights = AmbientLights(ambient_color=(1.0, 1.0, 1.0), device=device)
    else:
        raise ValueError(f"Unknown lighting scheme: {lighting}. Use 'directional' or 'ambient' or implement your own.")

    raster_settings = RasterizationSettings(
        image_size=img_size,
        bin_size=None,
        max_faces_per_bin=None
    )
    blend_params = BlendParams(background_color=torch.zeros(3))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params
        )
    )

    for j, mesh in enumerate(batch_mesh_data):
        meshes = mesh.extend(num_proj)
        num_faces = mesh.num_faces_per_mesh()
        # to avoid memory overflow while rendering meshes with too many faces. Higher max_faces_per_bin
        # means more memory usage for rendering
        raster_settings.max_faces_per_bin = num_faces // 2 if num_faces >= 23000 else None
        images = renderer(meshes, raster_settings=raster_settings)  # shape [num_projections, H, W, 4], 4:RGBA
        list_images.append(images)

    return torch.stack(list_images, dim=0)  # shape [batch_size, num_projections, H, W, 4], 4:RGBA


def render_feature_projections(
    batch_graph: Batch,
    batch_mesh_data: Meshes,
    img_size: int,
    num_proj: int,
    elev: torch.Tensor,
    azim: torch.Tensor,
    dist: float
) -> torch.Tensor:
    """
    Renders feature-based projections of a batch of meshes, where vertex features are used
    as vertex colors to visualize learned geometric attributes.

    Args:
        batch_graph (Batch): A PyTorch Geometric `Batch` object containing vertex features (`x`).
        batch_mesh_data (Meshes): A batch of PyTorch3D `Meshes`.
        img_size (int): Desired height and width of the rendered image.
        num_proj (int): Number of viewpoints per mesh.
        elev (torch.Tensor): Tensor of shape [num_proj] specifying elevation angles.
        azim (torch.Tensor): Tensor of shape [num_proj] specifying azimuth angles.
        dist (float): Distance from camera to object.

    Returns:
        torch.Tensor: Tensor of shape [batch_size, num_proj, H, W, feature_dim]
            representing rendered projections with feature-based colorization.
    """
    device = batch_mesh_data.device
    feature_dim = batch_graph.x.shape[-1]
    list_images = []

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    lights = AmbientLights(device=device, ambient_color=torch.ones(size=(1, feature_dim)))

    raster_settings = RasterizationSettings(
        image_size=img_size,
        bin_size=None,
        max_faces_per_bin=None
    )

    materials = Materials(
        device=device,
        ambient_color=torch.ones(size=(1, feature_dim)),
        specular_color=torch.zeros(size=(1, feature_dim)),
        diffuse_color=torch.zeros(size=(1, feature_dim))
    )
    blend_params = BlendParams(background_color=torch.zeros(feature_dim))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=blend_params
        )
    )

    for j, (graph, mesh) in enumerate(zip(batch_graph.to_data_list(), batch_mesh_data)):
        verts_features = graph.x.unsqueeze(dim=0)  # shape [1, num_vertices, feature_dim]
        textures = TexturesVertex(verts_features=verts_features.to(device))
        mesh.textures = textures
        meshes = mesh.extend(num_proj)
        num_faces = mesh.num_faces_per_mesh()
        # to avoid memory overflow while rendering meshes with too many faces. Higher max_faces_per_bin
        # means more memory usage for rendering
        raster_settings.max_faces_per_bin = num_faces if num_faces >= 20000 else None
        images = renderer(meshes, raster_settings=raster_settings)[..., :-1]  # drop alpha channel
        list_images.append(images)

    return torch.stack(list_images, dim=0)  # shape [batch_size, num_projections, H, W, feature_dim]


def render_projections(
    batch_graph: Batch,
    batch_mesh_data: Meshes,
    num_proj: int,
    training: bool,
    img_size: int = 256,
    angle_aug: bool = False,
    lighting: str = 'directional'
) -> List[torch.Tensor]:
    """
    Renders both RGB and feature projections for a batch of 3D meshes from multiple viewpoints.

    Args:
        batch_graph (Batch): Batched graph data containing vertex features.
        batch_mesh_data (Meshes): Batch of 3D mesh objects to render.
        num_proj (int): Number of viewpoints per mesh.
        training (bool): Whether in training mode (enables random viewpoint sampling and augmentation).
        img_size (int, optional): Image size for RGB projections. Defaults to 256.
        angle_aug (bool, optional): If True, apply Gaussian noise to camera angles for augmentation. Defaults to False.
        lighting (str, optional): Lighting scheme, either 'directional' or 'ambient'. Defaults to 'directional'.

    Returns:
        List[torch.Tensor]: A list containing:
            - Tensor of shape [B, num_proj, H, W, 4] for RGB projections (RGBA).
            - Tensor of shape [B, num_proj, H', W', C] for feature projections.
    """
    if training:
        # augmentation on viewpoint selections
        p = torch.tensor([1 / 6] * 6)
        all_elev = torch.tensor([0.0, 0.0, 0.0, 0.0, 90.0, 270.0])
        all_azim = torch.tensor([0.0, 90.0, 180.0, 270.0, 0.0, 0.0])
        sel_ind = p.multinomial(num_samples=num_proj, replacement=False)
        elev = all_elev[sel_ind]
        azim = all_azim[sel_ind]
        # angle augmentation
        if angle_aug:
            elev = torch.normal(mean=elev, std=torch.tensor([22.5] * num_proj))
            azim = torch.normal(mean=azim, std=torch.tensor([22.5] * num_proj))
    else:
        elev = torch.tensor([0.0, 0.0, 0.0, 0.0, 90.0, 270.0])
        azim = torch.tensor([0.0, 90.0, 180.0, 270.0, 0.0, 0.0])
        num_proj = 6

    # projection distance
    dist = 2.7
    # shape [batch_size, num_projections, H, W, 4], 4:RGBA
    rgb_projections = render_rgb_projections(
        batch_mesh_data=batch_mesh_data,
        img_size=img_size,
        num_proj=num_proj,
        elev=elev,
        azim=azim,
        lighting=lighting,
        dist=dist
    )
    # visually verify validity of projections
    # for mesh_proj in rgb_projections:
    #     plot_projections(mesh_proj, rows=2, cols=3)

    # shape [batch_size, num_projections, H, W, feature_dim]
    feat_size = img_size // 4
    feature_projections = render_feature_projections(
        batch_graph=batch_graph,
        batch_mesh_data=batch_mesh_data,
        img_size=feat_size,
        num_proj=num_proj,
        elev=elev,
        azim=azim,
        dist=dist
    )
    # visually verify validity of projections
    # for mesh_proj in feature_projections:
    #     plot_projections(mesh_proj, rows=2, cols=3)

    # free up memory
    # torch.cuda.empty_cache()
    projections = [rgb_projections, feature_projections]

    return projections


def send_dict_tensors_to_device(
    dict_of_tns: Dict,
    device: torch.device
) -> Dict:
    """
    Transfers all tensors in a dictionary to the specified device.

    Args:
        dict_of_tns (Dict): A dictionary mapping keys to different objects.
        device (torch.device): The device to which the objects will be moved (e.g., 'cuda' or 'cpu').

    Returns:
        Dict: A new dictionary with all objects moved to the specified device.
    """
    return {key: value.to(device) for key, value in dict_of_tns.items()}


def plot_projections(projection: torch.Tensor, rows: int, cols: int) -> None:
    """Plots first three channels (last dimension) of the input projection.

    Args:
        projection (torch.Tensor): projection tensor of shape [projection, H, W, feature_dim]
        rows (int): number of rows of subplot
        cols (int): number of cols of subplot
    """
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))

    for ax, im in zip(axs.ravel(), projection):
        ax.imshow(convert_range(im[..., :3].squeeze().cpu().detach().numpy()))
        # ax.set_axis_off()

    fig.savefig("dummy_fig.png")


def convert_range(img: np.ndarray) -> np.ndarray:
    """
    Normalizes the input image array to the range [0, 1].

    Args:
        img (np.ndarray): Input image array with arbitrary numeric range.

    Returns:
        np.ndarray: Normalized image array with values scaled to [0, 1].
    """
    img = np.double(img)
    max_val, min_val = img.max(), img.min()

    return (img - min_val) / (max_val - min_val)


def accumulate_step_outputs(
    step_outputs: Dict,
    names_batched: List[str],
    pred_quality: torch.Tensor,
    sample_batched: Dict,
    reg_loss: torch.Tensor,
    rank_loss: torch.Tensor,
    loss: torch.Tensor
) -> Dict:
    """
    Accumulates step outputs (predictions, losses, targets) into a running dictionary
    during training or evaluation over a batch of samples.

    Args:
        step_outputs (Dict): Dictionary storing accumulated values.
        names_batched (List[str]): List of sample identifiers in the current batch.
        pred_quality (torch.Tensor): Predicted quality scores for the batch.
        sample_batched (Dict): Dictionary containing the ground-truth target scores.
        reg_loss (torch.Tensor): Regression loss for the batch.
        rank_loss (torch.Tensor): Ranking loss for the batch.
        loss (torch.Tensor): Total loss for the batch.

    Returns:
        Dict: Updated `step_outputs` dictionary with the current batch results appended.
    """
    step_outputs['names'] += names_batched
    step_outputs['preds'] = torch.cat((step_outputs['preds'], pred_quality), dim=0)
    step_outputs['targets'] = torch.cat((step_outputs['targets'], sample_batched['score']), dim=0)
    step_outputs['reg_loss'] = torch.cat((step_outputs['reg_loss'], reg_loss.unsqueeze(0) * len(names_batched)), dim=0)
    step_outputs['rank_loss'] = torch.cat((step_outputs['rank_loss'],
                                           rank_loss.unsqueeze(0) * len(names_batched)), dim=0)
    step_outputs['loss'] = torch.cat((step_outputs['loss'], loss.unsqueeze(0) * len(names_batched)), dim=0)

    return step_outputs


def count_model_params(model: torch.nn.Module) -> str:
    """
    Generates a detailed summary of the number of trainable parameters in a model,
    broken down by submodules.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        str: A formatted string summarizing parameter counts per submodule and the total count.
    """
    output_string = "\n"
    submodel_dict = {}
    output_string += (100 * "=") + "\n"
    output_string += (f"{model.__class__.__name__:<{60}} Num Params") + "\n"
    output_string += (100 * "=") + "\n"
    for name, param in model.named_parameters():
        if param.requires_grad:
            output_string += (f"{name:<{60}} {param.numel()}") + "\n"
            if name.split('.')[0] not in submodel_dict.keys():
                submodel_dict[name.split('.')[0]] = []
            submodel_dict[name.split('.')[0]].append(param.numel())
    output_string += (100 * "=") + "\n"
    total = 0
    for key, value in submodel_dict.items():
        output_string += (f"{key:<{60}} {sum(value)}") + "\n"
        total += sum(value)
    output_string += (100 * "=") + "\n"
    output_string += (f"{str('Total'):<{60}} {total}")

    return output_string


def create_pred_dataframe(main_df: pd.DataFrame, epoch_outputs: Dict, epoch: int) -> pd.DataFrame:
    """
    Updates the main DataFrame with prediction results from a specific epoch.
    Adds predicted values and error columns corresponding to the epoch.

    Args:
        main_df (pd.DataFrame): DataFrame tracking predictions and targets over epochs.
        epoch_outputs (Dict): Dictionary containing 'names', 'targets', and 'preds'.
        epoch (int): Current epoch index.

    Returns:
        pd.DataFrame: Updated DataFrame with new columns for predictions and errors for the given epoch.

    Raises:
        Warning: If the stimulus ordering does not match between `main_df` and new outputs.
    """
    temp_df = pd.DataFrame(columns=['Stimulus', 'Target', 'Pred'])
    temp_df['Stimulus'] = epoch_outputs['names']
    temp_df['Target'] = epoch_outputs['targets'].detach().cpu().numpy()
    temp_df['Pred'] = epoch_outputs['preds'].detach().cpu().numpy()
    temp_df.sort_values(by=['Stimulus'], ignore_index=True, inplace=True)

    if main_df['Stimulus'].empty:
        main_df['Stimulus'] = temp_df['Stimulus']
        main_df['Target'] = temp_df['Target']

    if all(main_df['Stimulus'] == temp_df['Stimulus']):
        main_df[f'Epoch{epoch + 1}-pred'] = temp_df['Pred']
        main_df[f'Epoch{epoch + 1}-err'] = temp_df['Target'] - temp_df['Pred']
    else:
        raise Warning('Dataframes are not alligned! Skipping adding the recent epoch outputs!')
    return main_df


def plcc(pred: torch.Tensor, target: torch.Tensor, mapping: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the Pearson Linear Correlation Coefficient (PLCC) between predictions and targets.
    Optionally applies a 5-parameter logistic mapping before computing the correlation.

    Args:
        pred (torch.Tensor): Predicted quality scores (shape: [N]).
        target (torch.Tensor): Ground-truth quality scores (shape: [N]).
        mapping (bool, optional): If True, apply a logistic function to fit `pred` to `target`. Defaults to False.
        eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Scalar tensor representing the PLCC score.
    """
    if not mapping:
        pred = pred - torch.mean(pred)
        target = target - torch.mean(target)
        plcc = torch.sum(pred * target) / (torch.sqrt(torch.sum(pred ** 2)) * torch.sqrt(torch.sum(target ** 2)) + eps)
    else:
        x = pred.detach().cpu().numpy()
        y = target.detach().cpu().numpy()
        corr, _ = pearsonr(x, y)
        if corr > 0:
            c = np.mean(x)
            a = np.abs(np.max(y) - np.min(y))
            d = np.mean(y)
            b = 1 / np.std(x)
            e = 1
        else:
            c = np.mean(x)
            a = - np.abs(np.max(y) - np.min(y))
            d = np.mean(y)
            b = 1 / np.std(x)
            e = 1

        try:
            popt, _ = curve_fit(logistic_func, x, y, p0=[a, b, c, d, e])
            a = popt[0]
            b = popt[1]
            c = popt[2]
            d = popt[3]
            e = popt[4]
        except RuntimeError:
            pass

        x_log = logistic_func(x, a, b, c, d, e)
        plcc, _ = pearsonr(x_log, y)
        plcc = torch.tensor(plcc, device=pred.device)

    return plcc


def logistic_func(x: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """
    Five-parameter logistic function used to fit prediction scores to target scores.

    Args:
        x (np.ndarray): Input array (typically prediction values).
        a (float): Amplitude coefficient.
        b (float): Slope coefficient.
        c (float): Inflection point (center).
        d (float): Offset/baseline value.
        e (float): Linear term coefficient.

    Returns:
        np.ndarray: Transformed output after applying the logistic function.
    """
    x_ = np.array(x)
    tmp = 0.5 - 1 / (1 + np.exp(b * (x_ - c)))
    y = a * tmp + d + e * x_
    return y


def rank_array(input_array: torch.Tensor) -> torch.Tensor:
    """
    Computes ranks of elements in the input tensor.

    Args:
        input_array (torch.Tensor): 1D tensor of values to be ranked.

    Returns:
        torch.Tensor: 1D tensor of the same shape containing rank indices.
    """
    sorted_indices = torch.argsort(input_array)
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(len(input_array), device=input_array.device)

    return ranks


def srcc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the Spearman Rank Correlation Coefficient (SRCC) between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted scores (1D tensor or [N, 1]).
        target (torch.Tensor): Ground-truth scores (1D tensor or [N, 1]).

    Returns:
        torch.Tensor: Scalar tensor representing the SRCC score.
    """
    pred = pred.squeeze()
    target = target.squeeze()
    N = pred.size(0)

    pred_ranks = rank_array(pred)
    target_ranks = rank_array(target)

    num = 6 * torch.sum((pred_ranks - target_ranks) ** 2)
    denom = N * (N ** 2 - 1.0)
    srcc = 1 - num / denom

    return srcc


def generate_mask(input_tesnor: torch.Tensor, background_intensity: float = 0.0) -> torch.Tensor:
    """
    Generates a map that masks the background in the rendered viewpoints.
    The generated map contains 1's at non-background pixels and 0's at
    background pixels.

    Args:
        input_tesnor (torch.Tensor): Input tensor of shape [..., H, W, C], typically an image or rendering.
        background_intensity (float, optional): Intensity value representing the background. Defaults to 0.0.
    Returns:
        torch.Tensor: mask tensor of shape [..., H, W, C] that contains 1's at
                      non-background pixels and 0's at background pixels.
    """
    mask = ~(input_tesnor == background_intensity)

    return mask


def patchify_and_generate_mask(
    projections: Tuple[torch.Tensor, torch.Tensor],
    patch_size: int,
    background_intensity: float = 0.0,
    nonempty_ratio: float = 0.01,
    stride_ratio: float = 1.0
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Patchifies both rgb and feature projections, generates their bakground mask
    and discards empty patches.

    Args:
        projections (Tuple[torch.Tensor, torch.Tensor]): [rgb, feature] projections:
            rgb should be of shape [N, P, H, W, C1].
            feature should be of shape [N, P, H, W, C2].
        patch_size (int): size of patches
        background_intensity (float, optional): background intensity values. Defaults to 0.0.
        nonempty_ratio (float, optional): threshold ratio to consider a patch as nonempty. Defaults to 0.01.
        stride_ratio (float, optional): stride ratio for unfolding (patchifying) operation. Defaults to 1.0.

    Returns:
        Tuple[
            Tuple[torch.Tensor, torch.Tensor]],            # (patched RGB, patched features)
            torch.Tensor,                                  # non-empty patch indicator: [N*P, num_patches]
            torch.Tensor                                   # background mask: [num_nonempty_patches, Ps, Ps, 1]
        ]
    """
    # reshape projections
    N, P, H, W, C1 = projections[0].shape
    rgb_proj = projections[0].view(N * P, H, W, C1).permute(0, 3, 1, 2)  # output shape = [N*P, C1, H, W]
    N, P, H, W, C2 = projections[1].shape
    feature_proj = projections[1].view(N * P, H, W, C2).permute(0, 3, 1, 2)  # output shape = [N*P, C2, H, W]

    # patchify projections
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=int(patch_size * stride_ratio))
    # output shape = [N*P, num_patches, C1, Ps, Ps]
    patch_rgb_proj = unfold(rgb_proj).view(N * P, C1, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
    num_patches = patch_rgb_proj.shape[1]

    fpatch_size = patch_size // 4
    funfold = torch.nn.Unfold(kernel_size=fpatch_size, stride=int(fpatch_size * stride_ratio))
    # output shape = [N*P, num_patches, C2, Ps, Ps]
    patch_feature_proj = funfold(feature_proj).view(N * P, C2, fpatch_size, fpatch_size, -1).permute(0, 4, 1, 2, 3)

    # generate mask
    # output shape = [N*P, num_patches, Ps, Ps]
    ref_patch_rgb = patch_rgb_proj.view(N, P, num_patches, C1, patch_size, patch_size)[::2, :, :, :3, :, :]
    background_mask = ref_patch_rgb.any(dim=3) != background_intensity
    background_mask = torch.repeat_interleave(background_mask, repeats=2, dim=0)
    background_mask = background_mask.view(N*P, num_patches, patch_size, patch_size)

    # when less than nonempty_ratio is empty, consider the patch as empty, output shape = [N*P, num_patches]
    min_patches = 10
    background_mask_sum = torch.sum(background_mask, dim=(-1, -2)).view(N, P, num_patches)
    nonempty_patches = background_mask_sum >= ((patch_size ** 2) * nonempty_ratio)
    # find models that had less than "min_patches" valid patches
    model_indices = torch.nonzero(torch.sum(nonempty_patches, dim=(-1, -2)) < min_patches, as_tuple=True)[0]
    # select "min_patches" patches with the highest non-empty ratio for both reference and test models
    for idx in model_indices[::2]:
        patch_val, _ = torch.topk(background_mask_sum[idx].flatten(), k=min_patches)
        nonempty_patches[idx] = background_mask_sum[idx] > max(patch_val[-1] - 1, 0)  # refence model
        nonempty_patches[idx+1] = background_mask_sum[idx+1] > max(patch_val[-1] - 1, 0)  # test model
    nonempty_patches = nonempty_patches.view(N*P, num_patches)
    # output shape = [num_nonempty_patches, Ps, Ps, 1]
    background_mask = background_mask[nonempty_patches, :, :].unsqueeze(-1)

    # discard empty patches
    # output shape = [num_nonempty_patches, Ps, Ps, C1]
    patch_rgb_proj = patch_rgb_proj[nonempty_patches, :, :, :].permute(0, 2, 3, 1)
    # output shape = [num_nonempty_patches, Ps, Ps, C2]
    patch_feature_proj = patch_feature_proj[nonempty_patches, :, :, :].permute(0, 2, 3, 1)
    # patch_background_mask = background_mask[nonempty_patches, :, :]  # output shape = [num_nonempty_patches, Ps, Ps]

    # verification
    # for proj_idx in range(0, 24):
    #     if proj_idx % 6 == 0:
    #         plt.figure()
    #         plt.imshow(projections[0][0, proj_idx, :, :, :3].cpu().squeeze().detach())
    #         plt.savefig(f'dummy_1_{proj_idx}.png')

    #         plt.figure()
    #         plt.imshow(projections[0][1, proj_idx, :, :, :3].cpu().squeeze().detach())
    #         plt.savefig(f'dummy_2_{proj_idx}.png')

    #         plt.figure()
    #         abs_feat_diff = patch_feature_diff[0, proj_idx, :, :, :3])
    #         plt.imshow(convert_range(abs_feat_diff.cpu().squeeze().detach()))
    #         plt.savefig(f'dummy_3_{proj_idx}.png')

    #     sum_mat = torch.sum(nonempty_patches, dim=-1)
    #     low_range = high_range if proj_idx > 0 else 0
    #     high_range = low_range + sum_mat[proj_idx].item()
    #     plt.figure()
    #     for img_idx, patch_idx in enumerate(range(low_range, high_range)):
    #         plt.subplot(4, 4, img_idx + 1)
    #         plt.imshow(patch_rgb_proj[patch_idx, :, :, :3].cpu().squeeze().detach())
    #     plt.savefig(f'dummy_1.png')

    #     plt.figure()
    #     for img_idx, patch_idx in enumerate(range(low_range, high_range)):
    #         plt.subplot(4, 4, img_idx + 1)
    #         plt.imshow(patch_feature_diff[patch_idx, :, :, :3].cpu().squeeze().detach())
    #     plt.savefig(f'dummy_2.png')

    #     plt.figure()
    #     for img_idx, patch_idx in enumerate(range(low_range, high_range)):
    #         plt.subplot(4, 4, img_idx + 1)
    #         plt.imshow(patch_background_mask[patch_idx, ...].cpu().squeeze().detach())
    #     plt.savefig(f'dummy_3_{proj_idx}.png')
    #     plt.show()

    # torch.cuda.empty_cache()

    return ((patch_rgb_proj, patch_feature_proj), nonempty_patches, background_mask)


def pool_patch_features(
    N: int,
    num_proj: int,
    reg_feat: torch.Tensor,
    graph_feat: torch.Tensor,
    nonempty_patches: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    pools per-model regressor and graph features
    based on the non-empty patch indicator tensor.

    Args:
        N (int): number of models
        num_proj (int): number of projections (viewpoints) for each model
        reg_feat (torch.Tensor): regressor output features
            of shape [num_nonempty_patches, regressor_out_dim]
        graph_feat (torch.Tensor): graph pooled features
            of shape [num_nonempty_patches, graph_feat_out_dim]
        nonempty_patches (torch.Tensor): non-empty patch indicator tensor
            of shape [N*P, num_patches]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Pooled regressor features of shape [N, regressor_out_dim].
            - Pooled graph features of shape [N, graph_feat_out_dim].
    """
    # find patch index boundries for each model
    per_sample_patchmap = torch.sum(nonempty_patches.view(N, num_proj, nonempty_patches.shape[1]), dim=(-1, -2))
    patch_boundries = torch.cumsum(per_sample_patchmap, dim=0)

    # average patch features - first model
    pooled_reg_feat = torch.mean(reg_feat[0:patch_boundries[0], :], dim=0, keepdim=True)
    pooled_graph_feat = torch.mean(graph_feat[0:patch_boundries[0], :], dim=0, keepdim=True)
    # average patch features - other models
    for idx, up_bound in enumerate(patch_boundries[1:]):
        low_bound = patch_boundries[idx]
        model_reg_feat = torch.mean(reg_feat[low_bound:up_bound, :], dim=0, keepdim=True)
        pooled_reg_feat = torch.cat((pooled_reg_feat, model_reg_feat), dim=0)
        model_graph_feat = torch.mean(graph_feat[low_bound:up_bound, :], dim=0, keepdim=True)
        pooled_graph_feat = torch.cat((pooled_graph_feat, model_graph_feat), dim=0)

    return pooled_reg_feat, pooled_graph_feat


def apply_hflip_and_vflip(
    projections: Tuple[torch.Tensor, torch.Tensor],
    bg_mask: torch.Tensor,
    p: float = 0.5
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Applies Horizontal and Vertical Flip Augmentations (including the randomization)
    to inputs.

    Args:
        projections (Tuple[torch.Tensor, torch.Tensor]): Tuple containing:
            - RGB projections of shape [N, Ps, Ps, C1]
            - Feature projections of shape [N, Ps//4, Ps//4, C2], or None
        bg_mask (torch.Tensor): Background mask of shape [N, Ps, Ps, 1].
        p (float, optional): Probability of applying each flip. Defaults to 0.5.

    Returns:
        Tuple:
            - Tuple[torch.Tensor, torch.Tensor]: Flipped RGB and feature projections.
            - torch.Tensor: Flipped background mask.
    """
    rgb = projections[0].permute(0, 3, 1, 2)  # shape [N, C1, Ps, Ps]
    bg_mask = bg_mask.permute(0, 3, 1, 2)  # shape [N, 1, Ps, Ps]
    feat = projections[1].permute(0, 3, 1, 2)
    # hflip
    if torch.rand(1) < p:
        rgb = hflip(rgb)
        bg_mask = hflip(bg_mask)
        feat = hflip(feat)
    # vflip
    if torch.rand(1) < p:
        rgb = vflip(rgb)
        bg_mask = vflip(bg_mask)
        feat = vflip(feat)
    rgb = rgb.permute(0, 2, 3, 1)  # shape [num_nonempty_patches, Ps, Ps, C1]
    bg_mask = bg_mask.permute(0, 2, 3, 1)  # shape [num_nonempty_patches, Ps, Ps, 1]
    feat = feat.permute(0, 2, 3, 1)
    projections = [rgb, feat]

    return projections, bg_mask


class RankLoss(torch.nn.Module):
    """
    Rank-based loss function that penalizes incorrect ordering of prediction pairs.
    Given two sets of predictions and targets, the loss encourages preserving relative ranking.

    [Ref] Wei Sun, Xiongkuo Min, Wei Lu, and Guangtao Zhai. A deep learning based no-reference quality assessment model
    for ugc videos. In Proceedings of the 30th ACM International Conference on Multimedia.
    """

    def __init__(self, **kwargs):
        super(RankLoss, self).__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the pairwise ranking loss.

        Args:
            preds (torch.Tensor): Predicted scores of shape [N] or [N, 1].
            targets (torch.Tensor): Ground-truth scores of shape [N] or [N, 1].

        Returns:
            torch.Tensor: Scalar loss tensor representing the average pairwise ranking loss.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        targets = targets.unsqueeze(0).repeat(n, 1)
        targets_t = targets.t()
        masks = torch.sign(targets - targets_t)
        exclude_self = (torch.abs(targets - targets_t) > 0)
        rank_loss = exclude_self * torch.relu(torch.abs(targets - targets_t) - masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (exclude_self.sum() + 1e-08)
        return rank_loss

import argparse
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes

from hybridmqa.data.data_io_utils import load_objs_as_meshes
from hybridmqa.model.archit import GNN, BaseEncoder, HybridMQA, QualityEncoder
from hybridmqa.utils import geo_map_interp, send_dict_tensors_to_device


def compute_normal_and_vertex_map(mesh: Meshes) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the 2D vertex position map and normal map from a given 3D mesh using UV mapping.

    Args:
        mesh (Meshes): A PyTorch3D `Meshes` object containing a single mesh instance.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - vertex_map: Tensor of shape [H, W, 3] representing 3D vertex positions in 2D UV space.
            - normal_map: Tensor of shape [H, W, 3] representing surface normals in 2D UV space.
    """
    vertices, face_verts = mesh.get_mesh_verts_faces(0)
    uvs = mesh.textures.verts_uvs_list()[0]
    face_uvs = mesh.textures.faces_uvs_list()[0]
    normals = mesh.verts_normals_packed()

    # change to numpy array
    vertices = vertices.cpu().numpy()
    uvs = uvs.cpu().numpy()
    normals = normals.cpu().numpy()
    face_verts = face_verts.cpu().numpy()
    face_uvs = face_uvs.cpu().numpy()

    # TODO: check UV values to be in the range [0, 1], o.w. raise an error with proper message
    # TODO: check the range of vertex_map and normal_map values
    vertex_map, normal_map = geo_map_interp(
        vertices,
        uvs,
        normals,
        face_verts,
        face_uvs,
        H=256,
        W=256,
        num_samples=50
    )
    vertex_map = torch.from_numpy(vertex_map).float()
    normal_map = torch.from_numpy(normal_map).float()
    # # assert vertex_map in range [-1, 1]
    # assert np.all(vertex_map >= -1.1) and np.all(vertex_map <= 1.1), \
    #     f'{vertex_map.min()}, {vertex_map.max()}'

    return vertex_map, normal_map


def load_meshes_and_2D_maps(ref_path: str, dist_path: str, vcmesh: bool = False) -> Dict[str, torch.Tensor | Meshes]:
    """
    Loads reference and distorted meshes and computes their 2D vertex and normal maps.

    Args:
        ref_path (str): Path to the reference mesh (.obj file).
        dist_path (str): Path to the distorted mesh (.obj file).
        vcmesh (bool, optional): If True, treats input meshes as vertex-color meshes. Defaults to False.

    Returns:
        Dict[str, torch.Tensor | Meshes]: A dictionary with the following keys:
            - 'mesh_data': A PyTorch3D `Meshes` object containing the loaded meshes.
            - 'input_concat': A tensor of shape [2, C, H, W] containing the stacked input maps
                              for reference and distorted meshes, or dummy zeros if `vcmesh` is True.
    """
    # load obj file with pytorch3d
    mesh_data = load_objs_as_meshes([ref_path, dist_path], vcmesh=vcmesh)

    # normalize vertex positions to be in the range [-1, 1]
    # NOTE: For 3D scenes (not objects) like 'the_serving_room' in the TSMD dataset, you should either
    #       change the normalization or adjust the camera distance to render the scene properly.
    bbox = mesh_data.get_bounding_boxes()
    max_dim, _ = torch.max(bbox[:, :, 1] - bbox[:, :, 0], dim=-1)
    mesh_data._verts_list[0] = 2 * (mesh_data._verts_list[0] - (bbox[0, :, 0] / 2 + bbox[0, :, 1] / 2)) / max_dim[0]
    mesh_data._verts_list[1] = 2 * (mesh_data._verts_list[1] - (bbox[1, :, 0] / 2 + bbox[1, :, 1] / 2)) / max_dim[1]

    if not vcmesh:
        # load texture maps
        # TODO: check if the texture maps exist
        texture_ref = mesh_data.textures._maps_list[0].clone().detach()
        texture_ref = F.interpolate(texture_ref.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear',
                                    align_corners=False).squeeze(0).permute(1, 2, 0)
        texture_dist = mesh_data.textures._maps_list[1].clone().detach()
        texture_dist = F.interpolate(texture_dist.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear',
                                     align_corners=False).squeeze(0).permute(1, 2, 0)
        # compute vertex and normal maps
        vertex_map_ref, normal_map_ref = compute_normal_and_vertex_map(mesh_data[0])
        vertex_map_dist, normal_map_dist = compute_normal_and_vertex_map(mesh_data[1])

        # Concatenate and transpose to (C, H, W)
        input_concat_ref = torch.cat((texture_ref, normal_map_ref, vertex_map_ref), dim=2).permute(2, 0, 1)
        input_concat_dist = torch.cat((texture_dist, normal_map_dist, vertex_map_dist), dim=2).permute(2, 0, 1)
    else:
        input_concat_ref, input_concat_dist = torch.zeros(1), torch.zeros(1)

    data_sample = {
        'mesh_data': mesh_data,
        'input_concat': torch.stack([input_concat_ref, input_concat_dist], dim=0)
    }
    return data_sample


def build_model(ckpt_path: str, device: torch.device, vcmesh: bool = False) -> HybridMQA:
    """
    Builds and initializes the HybridMQA model for mesh quality assessment.

    Args:
        ckpt_path (str): Path to the trained model checkpoint (.pth file).
        device (torch.device): Target device for model inference (e.g., torch.device("cuda")).
        vcmesh (bool, optional): If True, the model is configured for vertex-color meshes. Defaults to False.

    Returns:
        HybridMQA: The fully constructed and checkpoint-loaded model ready for inference.
    """
    # create Base Encoder
    base_enc = BaseEncoder(hidden_channels=32, out_channels=16).to(device)
    # create GNN
    gnn = GNN(in_channels=16 if not vcmesh else 9, hidden_channels=16, out_channels=16).to(device)
    # create Quality Encoder
    quality_enc = QualityEncoder(in_channels=16, hidden_channels=32, out_channels=64).to(device)
    # create HybridMQA model
    # NOTE: Renderer uses directonal lighting by default. You can change it to 'ambient' lighting or implement your own.
    model = HybridMQA(
        base_enc=base_enc,
        gnn=gnn,
        quality_enc=quality_enc,
        fc_input_dim=16 + 5 * 64,
        lighting='ambient',
        vcmesh=vcmesh
    ).to(device)

    # load the model checkpoint
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load reference and distorted meshes
    data_sample = load_meshes_and_2D_maps(args.ref_mesh, args.dist_mesh, vcmesh=args.vcmesh)
    data_sample = send_dict_tensors_to_device(dict_of_tns=data_sample, device=device)

    # create an instance of the model
    model = build_model(args.ckpt_path, device, vcmesh=args.vcmesh)

    with torch.no_grad():
        pred_quality = model(data_sample)

    print(f"Predicted Quality Score: {pred_quality:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with HybridMQA')
    parser.add_argument('--ref_mesh', type=str, required=True, help='Path to the reference mesh (.obj)')
    parser.add_argument('--dist_mesh', type=str, required=True, help='Path to the distorted mesh (.obj)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--vcmesh', action='store_true', help='raise if the input meshes are vertex-color meshes')
    args = parser.parse_args()

    main()

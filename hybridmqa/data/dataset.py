import os
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch3d.structures import join_meshes_as_batch
from skimage import io
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset

from hybridmqa.data.data_io_utils import load_objs_as_meshes


class TMQADataset(Dataset):
    """
    A PyTorch dataset class for loading and processing the SJTU-TMQA dataset.

    This dataset pairs reference and distorted 3D meshes with their corresponding multi-channel
    2D maps (texture, normals, vertex positions) and a subjective MOS value.

    Args:
        root_dir (str): Path to the root directory of the dataset.
        split (str, optional): Which data split to use. Options are "train", "test", or "all". Defaults to "train".
        normalize_score (bool, optional): If True, normalize the MOS values to range [0, 1]. Defaults to False.
        seed (Optional[int], optional): Random seed used for the 80/20 train/test split when `kfold_seed` is not set.
            if kfold_seed is not None, seed should be in range [0, 4]. Defaults to None.
        kfold_seed (Optional[int], optional): Enables 5-fold cross-validation.
            If set, the `seed` specifies the fold index (0–4). Defaults to None.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        normalize_score: bool = False,
        seed: Optional[int] = None,
        kfold_seed: Optional[int] = None
    ) -> None:
        self.root_dir = root_dir
        self.normalize_score = normalize_score
        self.ref_dir = os.path.join(self.root_dir, 'reference_dataset')
        self.dist_dir = os.path.join(self.root_dir, 'distortion_dataset')
        self.file = os.path.join(self.dist_dir, 'SJTU-TMQA_MOS.csv')
        self.data = pd.read_csv(self.file)

        source_models = self.data['model'].unique()
        if kfold_seed is None:
            # randomly shuffle the source models with a given seed
            if seed is not None:
                np.random.seed(seed)
                np.random.shuffle(source_models)
            source_splits = [source_models[:int(0.8 * len(source_models))],
                             source_models[int(0.8 * len(source_models)):]]
        else:  # switch to kfold
            # if kfold_seed is not None, seed should be in range [0, 4]: fold0 --> seed = 0, ..., fold4 --> seed = 4
            kfold = KFold(n_splits=5, shuffle=True, random_state=kfold_seed)
            for fold_ind, (train_ind, test_ind) in enumerate(kfold.split(source_models)):
                if fold_ind == seed:
                    source_splits = [source_models[train_ind], source_models[test_ind]]

        target_sources = {'train': source_splits[0], 'test': source_splits[1], 'all': source_models}
        self.data = self.data[self.data['model'].isin(target_sources[split])]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a single sample from the dataset given its index.

        Each sample consists of a reference mesh, a distorted mesh, corresponding multi-channel
        2D input maps (concatenated texture, normal, and vertex maps), a subjective MOS score, and the object name.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - 'mesh_data': A `Meshes` object with both reference and distorted mesh.
                - 'input_concat': List of two tensors (reference, distorted) with shape [9, 256, 256].
                - 'score': The MOS value as a float tensor.
                - 'name': Sample name as a string.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_name = self.data.iloc[idx]['name']
        model, distortion, level = sample_name.split('_')
        dist_filename = os.path.join(self.dist_dir, distortion, model, level, model)
        ref_filename = os.path.join(self.ref_dir, model, model)
        score = torch.tensor(self.data.iloc[idx]['MOS'], dtype=torch.float32)

        if self.normalize_score:
            # normalize the score to be in the range [0, 1], now is [0, 10]
            score = 1.0 - (score / 10.0)

        # load obj files
        mesh_data = load_objs_as_meshes([ref_filename + ".obj", dist_filename + ".obj"], load_tensor=True)
        # normalize vertex positions to be in the range [-1, 1]
        bbox = mesh_data.get_bounding_boxes()
        max_dim, _ = torch.max(bbox[:, :, 1] - bbox[:, :, 0], dim=-1)
        mesh_data._verts_list[0] = 2 * (mesh_data._verts_list[0] - (bbox[0, :, 0] / 2 + bbox[0, :, 1] / 2)) / max_dim[0]
        mesh_data._verts_list[1] = 2 * (mesh_data._verts_list[1] - (bbox[1, :, 0] / 2 + bbox[1, :, 1] / 2)) / max_dim[1]

        # Load image, normal map, and vertex map
        texture_ref = torch.from_numpy(io.imread(ref_filename + ".jpg")).float() / 255.0
        texture_ref = F.interpolate(texture_ref.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear',
                                    align_corners=False).squeeze(0).permute(1, 2, 0)
        texture_dist = torch.from_numpy(io.imread(dist_filename + ".jpg")).float() / 255.0
        texture_dist = F.interpolate(texture_dist.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear',
                                     align_corners=False).squeeze(0).permute(1, 2, 0)
        vertex_map_ref = torch.from_numpy(np.load(ref_filename + "_interp_norm_vertex_map.npy")).float()
        vertex_map_dist = torch.from_numpy(np.load(dist_filename + "_interp_norm_vertex_map.npy")).float()
        normal_map_ref = torch.from_numpy(np.load(ref_filename + "_interp_normal_map.npy")).float()
        normal_map_dist = torch.from_numpy(np.load(dist_filename + "_interp_normal_map.npy")).float()

        # Concatenate and transpose to (C, H, W)
        input_concat_ref = torch.cat((texture_ref, normal_map_ref, vertex_map_ref), dim=2).permute(2, 0, 1)
        input_concat_dist = torch.cat((texture_dist, normal_map_dist, vertex_map_dist), dim=2).permute(2, 0, 1)

        sample = {'mesh_data': mesh_data,
                  'input_concat': [input_concat_ref, input_concat_dist],
                  'score': score,
                  'name': sample_name}

        return sample


class YN2023Dataset(Dataset):
    """
    A PyTorch dataset class for loading and processing the Nehme et al. dataset.

    This dataset pairs reference and distorted 3D meshes with their corresponding multi-channel
    2D maps (texture, normals, vertex positions) and a subjective MOS value.

    Args:
        root_dir (str): Path to the root directory of the dataset.
        split (str, optional): Which data split to use. Options are "train", "test", or "all". Defaults to "train".
        normalize_score (bool, optional): If True, normalize the MOS values to range [0, 1]. Defaults to False.
        seed (Optional[int], optional): Random seed used for the 80/20 train/test split when `kfold_seed` is not set.
            if kfold_seed is not None, seed should be in range [0, 4]. Defaults to None.
        kfold_seed (Optional[int], optional): Enables 5-fold cross-validation.
            If set, the `seed` specifies the fold index (0–4). Defaults to None.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        normalize_score: bool = False,
        seed: Optional[int] = None,
        kfold_seed: Optional[int] = None
    ) -> None:
        self.root_dir = root_dir
        self.normalize_score = normalize_score
        self.data = pd.read_csv(os.path.join(self.root_dir, 'MOS+CI_3000stimuli.csv'))
        split_list = self.data['stimulus'].str.split('_')
        self.data['source'] = split_list.apply(lambda x: '_'.join(x[:-6]))
        self.data['simp'] = split_list.apply(lambda x: x[-6])
        self.data['qp'] = split_list.apply(lambda x: x[-5])
        self.data['qt'] = split_list.apply(lambda x: x[-4])
        self.data['res'] = split_list.apply(lambda x: x[-2])
        self.data['JPEG'] = split_list.apply(lambda x: x[-1][1:])

        source_models = self.data['source'].unique()
        # Graph-LPIPS train/test split for random_state == 1
        if kfold_seed is None:
            source_splits = train_test_split(source_models, train_size=0.8, random_state=seed, shuffle=True)
        else:  # switch to kfold
            # if kfold_seed is not None, seed should be in range [0, 4]: fold0 --> seed = 0, ..., fold4 --> seed = 4
            kfold = KFold(n_splits=5, shuffle=True, random_state=kfold_seed)
            for fold_ind, (train_ind, test_ind) in enumerate(kfold.split(source_models)):
                if fold_ind == seed:
                    source_splits = [source_models[train_ind], source_models[test_ind]]

        target_sources = {'train': source_splits[0], 'test': source_splits[1], 'all': source_models}
        self.data = self.data[self.data['source'].isin(target_sources[split])]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a single sample from the dataset given its index.

        Each sample consists of a reference mesh, a distorted mesh, corresponding multi-channel
        2D input maps (concatenated texture, normal, and vertex maps), a subjective MOS score, and the object name.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - 'mesh_data': A `Meshes` object with both reference and distorted mesh.
                - 'input_concat': List of two tensors (reference, distorted) with shape [9, 256, 256].
                - 'score': The MOS value as a float tensor.
                - 'name': Sample name as a string.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        dist_filename = os.path.join(
            self.root_dir,
            sample['source'],
            f'distortions/JPEG_resize{sample["res"]}_quality{sample["JPEG"]}',
            f'{sample["source"]}_{sample["simp"]}_{sample["qp"]}_{sample["qt"]}.obj'
        )
        ref_filename = os.path.join(self.root_dir, sample['source'], 'source', f'{sample["source"]}.obj')
        score = torch.tensor(sample['MOS'], dtype=torch.float32)

        if self.normalize_score:
            # normalize the score to be in the range [0, 1], now is [1, 5]
            score = 1.0 - (score - 1.0) / 4.0

        # load obj file with pytorch3d
        mesh_data = load_objs_as_meshes([ref_filename, dist_filename], load_tensor=True)
        # normalize vertex positions to be in the range [-1, 1]
        bbox = mesh_data.get_bounding_boxes()
        max_dim, _ = torch.max(bbox[:, :, 1] - bbox[:, :, 0], dim=-1)
        mesh_data._verts_list[0] = 2 * (mesh_data._verts_list[0] - (bbox[0, :, 0] / 2 + bbox[0, :, 1] / 2)) / max_dim[0]
        mesh_data._verts_list[1] = 2 * (mesh_data._verts_list[1] - (bbox[1, :, 0] / 2 + bbox[1, :, 1] / 2)) / max_dim[1]

        # load image, normal map, and vertex map
        texture_ref = mesh_data.textures._maps_list[0].clone().detach()
        texture_ref = F.interpolate(texture_ref.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear',
                                    align_corners=False).squeeze(0).permute(1, 2, 0)
        texture_dist = mesh_data.textures._maps_list[1].clone().detach()
        texture_dist = F.interpolate(texture_dist.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear',
                                     align_corners=False).squeeze(0).permute(1, 2, 0)
        vertex_map_ref = torch.from_numpy(np.load(ref_filename[:-4] + "_interp_norm_vertex_map.npy")).float()
        vertex_map_dist = torch.from_numpy(np.load(dist_filename[:-4] + "_interp_norm_vertex_map.npy")).float()
        normal_map_ref = torch.from_numpy(np.load(ref_filename[:-4] + "_interp_comp_normal_map.npy")).float()
        normal_map_dist = torch.from_numpy(np.load(dist_filename[:-4] + "_interp_comp_normal_map.npy")).float()

        # Concatenate and transpose to (C, H, W)
        input_concat_ref = torch.cat((texture_ref, normal_map_ref, vertex_map_ref), dim=2).permute(2, 0, 1)
        input_concat_dist = torch.cat((texture_dist, normal_map_dist, vertex_map_dist), dim=2).permute(2, 0, 1)

        data_sample = {
            'mesh_data': mesh_data,
            'input_concat': [input_concat_ref, input_concat_dist],
            'score': score,
            'name': sample['stimulus']
        }
        return data_sample


class VCMeshDataset(Dataset):
    """
    A PyTorch dataset class for loading and processing the CMDM dataset.

    This dataset pairs reference and distorted 3D meshes with their subjective MOS value.

    Args:
        root_dir (str): Path to the root directory of the dataset.
        split (str, optional): Which data split to use. Options are "train", "test", or "all". Defaults to "train".
        normalize_score (bool, optional): If True, normalize the MOS values to range [0, 1]. Defaults to False.
        seed (int, optional): Random seed used for the 80/20 train/test split. It chooses the test source model
            and should be an integer from 0 to 4.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        normalize_score: bool = False,
        seed: int = 2,
        **kwargs: Any
    ) -> None:
        self.root_dir = root_dir
        self.normalize_score = normalize_score
        self.data = pd.read_csv(os.path.join(self.root_dir, 'Per_Stimulus_MOS_and_MLE.csv'))
        source_models = self.data['source'].unique()
        train_split = np.delete(source_models, seed)
        test_split = [source_models[seed]]  # seed should be an integer from 0 to 4. Chooses the test source model.
        target_sources = {'train': train_split, 'test': test_split, 'all': source_models}
        self.data = self.data[self.data['source'].isin(target_sources[split])]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a single sample from the dataset given its index.

        Each sample consists of a reference mesh, a distorted mesh, a subjective MOS score, and the object name.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - 'mesh_data': A `Meshes` object with both reference and distorted mesh.
                - 'input_concat': dummy tensors.
                - 'score': The MOS value as a float tensor.
                - 'name': Sample name as a string.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        root_name = os.path.join(self.root_dir, 'Stimuli_Format_Obj', sample['source'])
        dist_filename = os.path.join(root_name, f'{sample["Stimuli"]}.obj')
        ref_filename = os.path.join(root_name, f'{sample["source"]}_Ref_0.obj')
        score = torch.tensor(sample['MOS'], dtype=torch.float32)

        if self.normalize_score:
            # normalize the score to be in the range [0, 1], now is [1, 5]
            score = 1.0 - (score - 1.0) / 4.0

        # load obj file with pytorch3d
        mesh_data = load_objs_as_meshes([ref_filename, dist_filename], load_tensor=True, vcmesh=True)
        # normalize vertex positions to be in the range [-1, 1]
        bbox = mesh_data.get_bounding_boxes()
        max_dim, _ = torch.max(bbox[:, :, 1] - bbox[:, :, 0], dim=-1)
        mesh_data._verts_list[0] = 2 * (mesh_data._verts_list[0] - (bbox[0, :, 0] / 2 + bbox[0, :, 1] / 2)) / max_dim[0]
        mesh_data._verts_list[1] = 2 * (mesh_data._verts_list[1] - (bbox[1, :, 0] / 2 + bbox[1, :, 1] / 2)) / max_dim[1]

        data_sample = {
            'mesh_data': mesh_data,
            'input_concat': [torch.zeros(1), torch.zeros(1)],
            'score': score,
            'name': sample['Stimuli']
        }
        return data_sample


class TSMDDataset(Dataset):
    """
    A PyTorch dataset class for loading and processing the TSMD dataset.

    This dataset pairs reference and distorted 3D meshes with their corresponding multi-channel
    2D maps (texture, normals, vertex positions) and a subjective MOS value.

    Args:
        root_dir (str): Path to the root directory of the dataset.
        split (str, optional): Which data split to use. Options are "train", "test", or "all". Defaults to "train".
        normalize_score (bool, optional): If True, normalize the MOS values to range [0, 1]. Defaults to False.
        seed (Optional[int], optional): Random seed used for the 80/20 train/test split when `kfold_seed` is not set.
            if kfold_seed is not None, seed should be in range [0, 4]. Defaults to None.
        kfold_seed (Optional[int], optional): Enables 5-fold cross-validation.
            If set, the `seed` specifies the fold index (0–4). Defaults to None.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        normalize_score: bool = False,
        seed: Optional[int] = None,
        kfold_seed: Optional[int] = None
    ) -> None:
        self.root_dir = root_dir
        self.normalize_score = normalize_score
        self.ref_dir = os.path.join(self.root_dir, 'Reference')
        self.dist_dir = os.path.join(self.root_dir, 'Distortion')
        self.file = os.path.join(self.root_dir, 'TSMD_MOS.xlsx')
        self.data = pd.read_excel(self.file)
        split_list = self.data['PVS'].str.split('_')
        self.data['source'] = split_list.apply(lambda x: '_'.join(x[:-4]))
        self.data['dec'] = split_list.apply(lambda x: x[-4])
        self.data['qp'] = split_list.apply(lambda x: x[-3])
        self.data['qt'] = split_list.apply(lambda x: x[-2])
        self.data['cqlevel'] = split_list.apply(lambda x: x[-1])

        # models that need uv fixing
        self.fix_uv_list = ['butterflies_collection', 'cyber_samurai', 'green_tree_frog', 'japanese_spiny_lobster',
                            'luna_lionfish', 'sakura_cherry_blossom']
        # scene models
        self.scene_models = ['the_great_drawing_room', 'the_serving_room']

        source_models = self.data['source'].unique()
        # rejecting models whose reference is under license and not publicaly available
        rejection_list = ['thomas_fr00170', 'mitch_fr00001', 'nathalie_fr00036']
        source_models = np.array([x for x in source_models if x not in rejection_list])

        if kfold_seed is None:
            # randomly shuffle the source models with a given seed
            if seed is not None:
                np.random.seed(seed)
                np.random.shuffle(source_models)
            source_splits = [source_models[:int(0.8 * len(source_models))],
                             source_models[int(0.8 * len(source_models)):]]
        else:  # switch to kfold
            # if kfold_seed is not None, seed should be in range [0, 4]: fold0 --> seed = 0, ..., fold4 --> seed = 4
            kfold = KFold(n_splits=5, shuffle=True, random_state=kfold_seed)
            for fold_ind, (train_ind, test_ind) in enumerate(kfold.split(source_models)):
                if fold_ind == seed:
                    source_splits = [source_models[train_ind], source_models[test_ind]]

        target_sources = {'train': source_splits[0], 'test': source_splits[1], 'all': source_models}
        self.data = self.data[self.data['source'].isin(target_sources[split])]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a single sample from the dataset given its index.

        Each sample consists of a reference mesh, a distorted mesh, corresponding multi-channel
        2D input maps (concatenated texture, normal, and vertex maps), a subjective MOS score, and the object name.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - 'mesh_data': A `Meshes` object with both reference and distorted mesh.
                - 'input_concat': List of two tensors (reference, distorted) with shape [9, 256, 256].
                - 'score': The MOS value as a float tensor.
                - 'name': Sample name as a string.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        dist_filename = os.path.join(
            self.dist_dir,
            sample['source'],
            f'{sample["PVS"][:-2]}_{sample["PVS"][-2:]}'
        )
        ref_filename = os.path.join(self.ref_dir, sample['source'], f'{sample["source"]}_C0-L5_deq_tri')
        score = torch.tensor(sample['MOS'], dtype=torch.float32)

        if self.normalize_score:
            # normalize the score to be in the range [0, 1], now is [1, 5]
            score = 1.0 - (score - 1.0) / 4.0

        # load obj file
        mesh_data = load_objs_as_meshes([ref_filename + ".obj", dist_filename + ".obj"], load_tensor=True)

        # fix uvs for some samples
        if sample['source'] in self.fix_uv_list:
            uvs = mesh_data.textures.verts_uvs_list()
            uv_ls = uvs[0][:, 0] > 1.0
            uvs[0][uv_ls, 0] = uvs[0][uv_ls, 0] - 1.0
            uv_ls = uvs[1][:, 0] > 1.0
            uvs[1][uv_ls, 0] = uvs[1][uv_ls, 0] - 1.0

        # normalize vertex positions to be in the range [-1, 1]
        vert_range = 15.43 if sample['source'] in self.scene_models else 2.0
        bbox = mesh_data.get_bounding_boxes()
        max_dim, _ = torch.max(bbox[:, :, 1] - bbox[:, :, 0], dim=-1)
        mesh_data._verts_list[0] = vert_range * (mesh_data._verts_list[0]
                                                 - (bbox[0, :, 0] / 2 + bbox[0, :, 1] / 2)) / max_dim[0]
        mesh_data._verts_list[1] = vert_range * (mesh_data._verts_list[1]
                                                 - (bbox[1, :, 0] / 2 + bbox[1, :, 1] / 2)) / max_dim[1]

        # Load image, normal map, and vertex map
        texture_ref = mesh_data.textures._maps_list[0].clone().detach()
        texture_ref = F.interpolate(texture_ref.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear',
                                    align_corners=False).squeeze(0).permute(1, 2, 0)
        texture_dist = mesh_data.textures._maps_list[1].clone().detach()
        texture_dist = F.interpolate(texture_dist.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear',
                                     align_corners=False).squeeze(0).permute(1, 2, 0)
        vertex_map_ref = torch.from_numpy(np.load(ref_filename + "_interp_norm_vertex_map.npy")).float()
        vertex_map_dist = torch.from_numpy(np.load(dist_filename + "_interp_norm_vertex_map.npy")).float()
        normal_map_ref = torch.from_numpy(np.load(ref_filename + "_interp_normal_map.npy")).float()
        normal_map_dist = torch.from_numpy(np.load(dist_filename + "_interp_normal_map.npy")).float()

        # Concatenate and transpose to (C, H, W)
        input_concat_ref = torch.cat((texture_ref, normal_map_ref, vertex_map_ref), dim=2).permute(2, 0, 1)
        input_concat_dist = torch.cat((texture_dist, normal_map_dist, vertex_map_dist), dim=2).permute(2, 0, 1)

        sample = {'mesh_data': mesh_data,
                  'input_concat': [input_concat_ref, input_concat_dist],
                  'score': score,
                  'name': sample['PVS']}

        return sample


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching mesh-based dataset samples.

    This function aggregates a list of individual sample dictionaries into a single batch dictionary.
    It handles batching of:
      - Mesh objects (as a `Meshes` batch using `join_meshes_as_batch`)
      - Tensor inputs (stacked as [2N, C, H, W], with reference/distorted alternation)
      - Scores, and sample names

    Args:
        batch (List[Dict[str, Any]]): A list of sample dictionaries returned from `__getitem__`.

    Returns:
        Dict[str, Any]: A dictionary with the following keys:
            - 'mesh_data': A batched `Meshes` object.
            - 'input_concat': A [2N, C, H, W] tensor of input maps.
            - 'score': A [N] tensor of MOS values.
            - 'names': A list of N sample names.
    """
    input_concats = []
    scores = []
    meshes = []
    names = []
    for item in batch:
        input_concats.extend(item['input_concat'])
        scores.append(item['score'])
        meshes.extend(item['mesh_data'])
        names.append(item['name'])

    # Combine the Meshes objects into a single batched Meshes object
    meshes_batch = join_meshes_as_batch(meshes)

    # Stack the tensor data
    input_concat_batch = torch.stack(input_concats, dim=0)
    score_batch = torch.stack(scores, dim=0)

    # Return a dictionary with all the batched data. In the batch dimension, odd indices correspond to distorted
    # inputs while even indices correspond to their reference counterparts.
    return {
        'mesh_data': meshes_batch,
        'input_concat': input_concat_batch,
        'score': score_batch,
        'names': names
    }


DATASET_REGISTRY: Dict[str, Type[Dataset]] = {
        'TMQA': TMQADataset,
        'YN2023': YN2023Dataset,
        'VCMesh': VCMeshDataset,
        'TSMD': TSMDDataset,
    }


class DatasetBuilder:
    """
    A  builder for loading supported datasets based on a name identifier.

    Supported names:
        - 'TMQA'
        - 'YN2023'
        - 'VCMesh'
        - 'TSMD'

    Example:
        >>> dataset = DatasetBuilder.build(
        >>>     name='TMQA',
        >>>     root_dir='/path/to/data',
        >>>     split='train',
        >>>     normalize_score=True,
        >>>     seed=0,
        >>>     kfold_seed=1
        >>> )
    """

    @staticmethod
    def build(name: str, **kwargs: Any) -> Dataset:
        """
        Builds and returns a dataset instance using the name identifier.

        Args:
            name (str): Name of the dataset (case-sensitive).
            **kwargs: All additional keyword arguments passed to the dataset constructor.

        Returns:
            Dataset: An instance of the selected dataset class.

        Raises:
            ValueError: If the dataset name is not supported.
        """
        dataset_cls = DATASET_REGISTRY.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unsupported dataset name '{name}'. Available options: {list(DATASET_REGISTRY.keys())}")
        return dataset_cls(**kwargs)

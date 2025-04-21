import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from iopath.common.file_io import PathManager
from pytorch3d.common.datatypes import Device
from pytorch3d.io.mtl_io import make_mesh_texture_atlas
from pytorch3d.io.obj_io import (_Aux, _Faces, _format_faces_indices,
                                 _load_materials, _parse_face)
from pytorch3d.io.utils import _make_tensor, _open_file
from pytorch3d.renderer import TexturesAtlas, TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes, join_meshes_as_batch

# NOTE: This script rewrites some functions of the obj_io.py script from pytorch3d to:
#   1. Add support for loading vertex-color meshes (vcmesh), which is not supported by pytorch3d 0.7.5.
#   2. Add support for loading and saving tensors of obj files (for better loading efficiency).


def _parse_obj(f, data_dir: str, vcmesh: bool = False):
    """
    Extended version of PyTorch3D's `_parse_obj` that optionally supports
    loading per-vertex colors in addition to standard OBJ content.

    For detailed documentation about the returned values and OBJ formatting,
    refer to the official PyTorch3D `obj_io.py::_parse_obj` function.

    Args:
        vcmesh (bool, optional): If True, parse vertex colors (R, G, B) in addition
            to vertex positions. Defaults to False.
    """
    verts, normals, verts_uvs = [], [], []
    faces_verts_idx, faces_normals_idx, faces_textures_idx = [], [], []
    faces_materials_idx = []
    material_names = []
    mtl_path = None
    verts_color = [] if vcmesh else None

    lines = [line.strip() for line in f]

    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    materials_idx = -1

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("mtllib"):
            if len(tokens) < 2:
                raise ValueError("material file name is not specified")
            # NOTE: only allow one .mtl file per .obj.
            # Definitions for multiple materials can be included
            # in this one .mtl file.
            mtl_path = line[len(tokens[0]):].strip()  # Take the remainder of the line
            mtl_path = os.path.join(data_dir, mtl_path)
        elif len(tokens) and tokens[0] == "usemtl":
            material_name = tokens[1]
            # materials are often repeated for different parts
            # of a mesh.
            if material_name not in material_names:
                material_names.append(material_name)
                materials_idx = len(material_names) - 1
            else:
                materials_idx = material_names.index(material_name)
        elif line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
            if vcmesh:
                vert_col = [float(x) for x in tokens[4:7]]
                if len(vert_col) != 3:
                    msg = "Vertex color %s does not have 3 values. Line: %s"
                    raise ValueError(msg % (str(vert), str(line)))
                verts_color.append(vert_col)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            verts_uvs.append(tx)
        elif line.startswith("vn "):  # Line is a normal.
            norm = [float(x) for x in tokens[1:4]]
            if len(norm) != 3:
                msg = "Normal %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(norm), str(line)))
            normals.append(norm)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            _parse_face(
                line,
                tokens,
                materials_idx,
                faces_verts_idx,
                faces_normals_idx,
                faces_textures_idx,
                faces_materials_idx,
            )

    return (
        verts,
        verts_color,
        normals,
        verts_uvs,
        faces_verts_idx,
        faces_normals_idx,
        faces_textures_idx,
        faces_materials_idx,
        material_names,
        mtl_path,
    )


def _load_obj_parse(
    f_obj,
    *,
    data_dir: str,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: PathManager,
    device: Device = "cpu",
    load_tensor: bool = False,
    save_tensor: bool = False,
    vcmesh: bool = False,
):
    """
    Extended version of PyTorch3D's `_load_obj` that supports loading and saving
    object tensors, as well as loading vertex-colored meshes.

    For full documentation on core functionality, refer to:
        `pytorch3d.io.obj_io._load_obj`

    Args:
        load_tensor (bool, optional): If True, load mesh data from a pre-saved `.pt` tensor file
            instead of parsing the `.obj` file. Defaults to False.
        save_tensor (bool. optional): If True, saves the parsed result as a `.pt` file to enable
            faster loading next time. Defaults to False.
        vcmesh (bool, optional): If True, enables support for vertex-colored meshes by parsing
            per-vertex RGB colors. Defaults to False.
    """

    if texture_wrap is not None and texture_wrap not in ["repeat", "clamp"]:
        msg = "texture_wrap must be one of ['repeat', 'clamp'] or None, got %s"
        raise ValueError(msg % texture_wrap)

    if not load_tensor:
        (
            verts,
            verts_color,
            normals,
            verts_uvs,
            faces_verts_idx,
            faces_normals_idx,
            faces_textures_idx,
            faces_materials_idx,
            material_names,
            mtl_path,
        ) = _parse_obj(f_obj, data_dir, vcmesh=vcmesh)

        verts = _make_tensor(verts, cols=3, dtype=torch.float32, device=device)  # (V, 3)
        verts_color = _make_tensor(verts_color, cols=3, dtype=torch.float32, device=device) if vcmesh else None
        normals = _make_tensor(
            normals,
            cols=3,
            dtype=torch.float32,
            device=device,
        )  # (N, 3)
        verts_uvs = _make_tensor(
            verts_uvs,
            cols=2,
            dtype=torch.float32,
            device=device,
        )  # (T, 2)

        faces_verts_idx = _format_faces_indices(
            faces_verts_idx, verts.shape[0], device=device
        )

        # Repeat for normals and textures if present.
        if len(faces_normals_idx):
            faces_normals_idx = _format_faces_indices(
                faces_normals_idx, normals.shape[0], device=device, pad_value=-1
            )
        if len(faces_textures_idx):
            faces_textures_idx = _format_faces_indices(
                faces_textures_idx, verts_uvs.shape[0], device=device, pad_value=-1
            )
        if len(faces_materials_idx):
            faces_materials_idx = torch.tensor(
                faces_materials_idx, dtype=torch.int64, device=device
            )
    else:
        load_path = os.path.splitext(f_obj.name)[0] + '.pt'
        obj_pt = torch.load(load_path)
        verts = obj_pt['verts']
        verts_color = obj_pt['verts_color'] if vcmesh else None
        normals = obj_pt['normals']
        verts_uvs = obj_pt['verts_uvs']
        faces_verts_idx = obj_pt['faces_verts_idx']
        faces_normals_idx = obj_pt['faces_normals_idx']
        faces_textures_idx = obj_pt['faces_textures_idx']
        faces_materials_idx = obj_pt['faces_materials_idx']
        material_names = obj_pt['material_names']
        mtl_path = os.path.join(data_dir, './' + obj_pt['mtl_path'].split('/')[-1]) if not vcmesh else None

    if save_tensor:
        save_path = os.path.splitext(f_obj.name)[0] + '.pt'
        if not vcmesh:
            torch.save({'verts': verts, 'normals': normals, 'verts_uvs': verts_uvs, 'faces_verts_idx': faces_verts_idx,
                        'faces_normals_idx': faces_normals_idx, 'faces_textures_idx': faces_textures_idx,
                        'faces_materials_idx': faces_materials_idx, 'material_names': material_names,
                        'mtl_path': mtl_path}, save_path)
        else:
            torch.save({'verts': verts, 'verts_color': verts_color, 'normals': normals, 'verts_uvs': verts_uvs,
                        'faces_verts_idx': faces_verts_idx, 'faces_normals_idx': faces_normals_idx,
                        'faces_textures_idx': faces_textures_idx, 'faces_materials_idx': faces_materials_idx,
                        'material_names': material_names, 'mtl_path': mtl_path}, save_path)

    texture_atlas = None
    if not vcmesh:
        material_colors, texture_images = _load_materials(
            material_names,
            mtl_path,
            data_dir=data_dir,
            load_textures=load_textures,
            path_manager=path_manager,
            device=device,
        )
    else:
        material_colors, texture_images = None, None

    if material_colors and not material_names:
        # usemtl was not present but single material was present in the .mtl file
        material_names.append(next(iter(material_colors.keys())))
        # replace all -1 by 0 material idx
        if torch.is_tensor(faces_materials_idx):
            faces_materials_idx.clamp_(min=0)

    if create_texture_atlas:
        # Using the images and properties from the
        # material file make a per face texture map.

        # Create an array of strings of material names for each face.
        # If faces_materials_idx == -1 then that face doesn't have a material.
        idx = faces_materials_idx.cpu().numpy()
        face_material_names = np.array(material_names)[idx]  # (F,)
        face_material_names[idx == -1] = ""

        # Construct the atlas.
        texture_atlas = make_mesh_texture_atlas(
            material_colors,
            texture_images,
            face_material_names,
            faces_textures_idx,
            verts_uvs,
            texture_atlas_size,
            texture_wrap,
        )

    faces = _Faces(
        verts_idx=faces_verts_idx,
        normals_idx=faces_normals_idx,
        textures_idx=faces_textures_idx,
        materials_idx=faces_materials_idx,
    )
    aux = _Aux(
        normals=normals if len(normals) else None,
        verts_uvs=verts_uvs if len(verts_uvs) else None,
        material_colors=material_colors,
        texture_images=texture_images,
        texture_atlas=texture_atlas,
    )
    return verts, faces, aux, verts_color


def load_obj(
    f,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    device: Device = "cpu",
    path_manager: Optional[PathManager] = None,
    load_tensor: bool = False,
    save_tensor: bool = False,
    vcmesh: bool = False
):
    """
    Wrapper around PyTorch3D's `load_obj` function with added support for:
      - vertex-color meshes
      - loading/saving cached tensor representations

    For full documentation on standard arguments and return values,
    refer to: `pytorch3d.io.obj_io.load_obj`.

    Args:
        load_tensor (bool, optional): If True, loads mesh data from a pre-saved `.pt` file
            instead of parsing the `.obj` file. Defaults to False.
        save_tensor (bool, optional): If True, saves the parsed result into a `.pt` file
            for faster future loading. Defaults to False.
        vcmesh (bool, optional): If True, enables parsing of vertex colors
            (R, G, B) along with positions. Defaults to False.
    """
    data_dir = "./"
    if isinstance(f, (str, bytes, Path)):
        # pyre-fixme[6]: For 1st argument expected `PathLike[Variable[AnyStr <:
        #  [str, bytes]]]` but got `Union[Path, bytes, str]`.
        data_dir = os.path.dirname(f)
    if path_manager is None:
        path_manager = PathManager()
    with _open_file(f, path_manager, "r") as f:
        return _load_obj_parse(
            f,
            data_dir=data_dir,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
            device=device,
            load_tensor=load_tensor,
            save_tensor=save_tensor,
            vcmesh=vcmesh
        )


def load_objs_as_meshes(
    files: list,
    device: Optional[Device] = None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: Optional[PathManager] = None,
    load_tensor: bool = False,
    save_tensor: bool = False,
    vcmesh: bool = False
):
    """
    Wrapper around PyTorch3D's `load_objs_as_meshes` that adds support for:
      - vertex-color meshes
      - loading from/saving to cached tensor (.pt) files

    For full documentation on mesh loading behavior, refer to:
        `pytorch3d.io.obj_io.load_objs_as_meshes`

    Args:
        load_tensor (bool, optional): If True, load mesh data from previously cached `.pt` files. Defaults to False.
        save_tensor (bool, optional): If True, save parsed mesh data to `.pt` files for reuse. Defaults to False.
        vcmesh (bool, optional): If True, supports loading vertex-color
            meshes using `TexturesVertex`. Defaults to False.
    """
    mesh_list = []
    for f_obj in files:
        verts, faces, aux, verts_color = load_obj(
            f_obj,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
            load_tensor=load_tensor,
            save_tensor=save_tensor,
            vcmesh=vcmesh
        )
        tex = None
        if create_texture_atlas:
            # TexturesAtlas type
            tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
        else:
            # TexturesUV type
            tex_maps = aux.texture_images
            if tex_maps is not None and len(tex_maps) > 0:
                verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
                faces_uvs = faces.textures_idx.to(device)  # (F, 3)
                image = list(tex_maps.values())[0].to(device)[None]
                tex = TexturesUV(
                    verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
                )
        if vcmesh:
            tex = TexturesVertex(verts_features=verts_color.unsqueeze(dim=0))
        mesh = Meshes(
            verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex
        )
        mesh_list.append(mesh)
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)

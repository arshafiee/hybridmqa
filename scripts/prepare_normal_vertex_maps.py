import argparse
import logging
import os

import numpy as np
from tqdm import tqdm

from hybridmqa.data.data_io_utils import load_objs_as_meshes
from hybridmqa.utils import geo_map_interp

# set up logging
logging.basicConfig(filename='./prepare_normal_map.log',
                    filemode='w', level=logging.INFO)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='prepare_map')
    parser.add_argument('--root_dir', type=str, help='root dir of the subset you want the prepare maps for')
    args = parser.parse_args()

    for subdir, dirs, files in os.walk(args.root_dir):
        for file in tqdm(files):
            if file.endswith('.obj'):

                obj_data = load_objs_as_meshes([os.path.join(subdir, file)])

                vertices, face_verts = obj_data[0].get_mesh_verts_faces(0)
                uvs = obj_data[0].textures.verts_uvs_list()[0]
                face_uvs = obj_data[0].textures.faces_uvs_list()[0]
                normals = obj_data[0].verts_normals_packed()

                # change to numpy array
                vertices = vertices.cpu().numpy()
                uvs = uvs.cpu().numpy()
                normals = normals.cpu().numpy()
                face_verts = face_verts.cpu().numpy()
                face_uvs = face_uvs.cpu().numpy()

                # NOTE: Comment this out if you are creating maps for any dataset other than TSMD!!
                # fix uvs for some samples
                uv_fix_list = ['butterflies_collection', 'cyber_samurai', 'green_tree_frog',
                               'japanese_spiny_lobster', 'luna_lionfish', 'sakura_cherry_blossom']
                if any(s in file for s in uv_fix_list) and ('TSMD' in args.root_dir):
                    uv_ls = uvs[:, 0] > 1.0
                    uvs[uv_ls, 0] = uvs[uv_ls, 0] - 1.0

                # normalize vertices and keep the aspect ratio
                max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max()
                vertices = 2 * (vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2) / max_range

                # if exists, skip
                if os.path.exists(os.path.join(subdir, file[:-4] + '_interp_normal_map.npy')):
                    logging.info('Skipping ' + os.path.join(subdir, file))
                    continue
                logging.info('Generating normal map for ' + os.path.join(subdir, file))

                # direct projection and rounding, using interpolation for better results
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

                # assert vertex_map in range [-1, 1]
                assert np.all(vertex_map >= -1.1) and np.all(vertex_map <= 1.1), \
                    f'{vertex_map.min()}, {vertex_map.max()}'

                # save vertex map to npy files
                np.save(os.path.join(subdir, file[:-4] + '_interp_norm_vertex_map.npy'), vertex_map)

                # save normal map to npy files
                np.save(os.path.join(subdir, file[:-4] + '_interp_normal_map.npy'), normal_map)

# Datasets

This folder instructions on how to obtain the datasets and how to organize them properly for use with our training and testing scripts.

---

## Dataset Preparation

Before beginning the training/testing process, you need to prepare normal and vertex maps for all the meshes of the
dataset:
```bash
python3 scripts/prepare_normal_vertex_maps.py /path/to/dataset/root/dir
```
This will:
- Recursively search the dataset directory for all `.obj` files.
- Generate vertex and normal maps for each mesh using UV mapping.
- Save the computed maps in NumPy format alongside the corresponding `.obj` file.

We currently support training and testing on four datasets:
- YN2023 (Nehmé et al.): [Dataset Link](https://yananehme.github.io/publications/2022-ACM-TOG)
- TMQA (SJTU-TMQA): [Dataset Link](https://ccccby.github.io/)
- TSMD: [Dataset Link](https://multimedia.tencent.com/resources/tsmd)
- VCMesh (CMDM): [Dataset Link](https://yananehme.github.io/publications/2020-IEEE-TVCG)

You can implement support for additional datasets by creating a custom dataset 
class in [`hybridmqa/data/dataset.py`](../hybridmqa/data/dataset.py) and start training/testing on them.

**Optionally**, to speed up training and testing, you can preprocess the dataset by iterating
through it once and saving mesh information as `.pt` tensors using the `save_tensor` argument
in the `load_objs_as_meshes` function 
(defined in [`hybridmqa/data/data_io_utils.py`](../hybridmqa/data/data_io_utils.py)).
This will store mesh data (e.g., vertices, faces, normals) in a more efficient format.
During the actual training or testing process, you can use the `load_tensor` argument
to load these `.pt` files instead of `.obj` files, significantly reducing loading time.

## Dataset Directory Structure

After downloading and extracting the datasets, please organize them in the following format. Please note refactorizations might be needed to align the dataset directories with the dataset classes (specifically for reading operations) as defined in [`hybridmqa/data/dataset.py`](../hybridmqa/data/dataset.py). Alternatively, you can modify the dataset classes to accommodate your dataset's structure.

### YN2023:
```
YN2023/                             # Root dir of the dataset
│
├── 1970.16_Neck_Amphora_100K/      # first object 
    ├── distortions/
    └── source/
├ ...
├── yourMesh/                       # last object
    ├── distortions/
    └── source/
└── MOS+CI_3000stimuli.csv
```
### TMQA:
```
TMQA/                             # Root dir of the dataset
│
├── reference_dataset/ 
    ├── airplane/                 # first object
    ├ ...
    └── zakopaneChair/            # last object
└── distortion_dataset/
    ├── ds/
    ├── gn/
    ├── JPEG/
    ├── qp/
    ├── qpqt/
    ├── qpqtJPEG/
    ├── simp/
        ├── airplane/             # first object
            ├── L01/
            ├ ...
            └── L05/
        ├ ...
        └── zakopaneChair/        # last object
    ├── simpNoTex/
    └── SJTU-TMQA_MOS.csv
```
### TSMD:
```
TSMD/                             # Root dir of the dataset
│
├── Reference/ 
    ├── apollo_11/                # first object
    ├ ...
    └── zakopane_chair/           # last object
└── Distortion/
    ├── apollo_11/                # first object
    ├ ...
    └── zakopane_chair/           # last object
└── TSMD_MOS.xlsx
```
### VCMesh:
```
VCMesh/                             # Root dir of the dataset
│
├── Stimuli_Format_Obj/ 
    ├── Aix/
    ├── Ari/
    ├── Chameleon/
    ├── Fish/
    └── Samurai/           
└── Per_Stimulus_MOS_and_MLE.csv
```
## Notes
Please note that:
- Meshes should be in `.obj` format.
- For textured meshes:
  - Each `.obj` file should reference its corresponding material `.mtl` file (typically specified at the top of the `.obj` file). Ensure the `.mtl` file is correctly placed and accessible.
  - Each `.mtl` file should reference the corresponding texture image. Ensure the texture image is correctly placed and accessible.
---

If you have questions or issues regarding dataset formatting or usage, feel free to open an issue in this repository.

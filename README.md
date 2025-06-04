<h1 align="center">HybridMQA: Exploring Geometry-Texture Interactions for Colored Mesh Quality Assessment (CVPR 2025)</h1>

<p align="center">
  <img src="docs/static/images/logo.png" alt="HybridMQA Logo" width="300"/>
</p>

<p align="center">
  <a href="https://arshafiee.github.io/hybridmqa/">🌐 Project Webpage</a> |
  <a href="https://arxiv.org/pdf/2412.01986">📄 Paper (arXiv)</a>
</p>

---

## Overview

**HybridMQA** is a learning-based framework for *full-reference quality assessment* of 3D colored meshes. It leverages both 3D and 2D modalities of 3D meshes and models complex interactions between their geometry and texture to assess their perceptual quality. Our method significantly outperforms existing techniques on multiple public datasets, aligning with human perception and generalizing better.

---

## Installation

Python ≥ 3.10 is required. Clone the repo and install dependencies via `pyproject.toml`:

```bash
git clone https://github.com/arshafiee/hybridmqa.git
cd hybridmqa
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Next, ensure that [PyTorch3D==0.7.5](https://github.com/facebookresearch/pytorch3d) is installed properly with CUDA support. We recommend using version `0.7.5` for best compatibility, but newer versions should also work. Follow their official [installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for setup or run:
```bash
MAX_JOBS=4 pip3 install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

---

## Usage

### Pretrained Weights

Pretrained weights are available on Hugging Face:

**[➡ Download from Hugging Face](https://huggingface.co/arshafiee/hybridmqa-checkpoint)**

You can also download the weights programmatically using the Hugging Face Hub API:

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="arshafiee/hybridmqa-checkpoint",
    filename="<checkpoint_name>.pth"
)
```

### Inference

To run quality prediction on a pair of reference and distorted 3D meshes:

```bash
python3 hybridmqa/run/inference.py \
  --ref_mesh path/to/reference.obj \
  --dist_mesh path/to/distorted.obj \
  --ckpt_path path/to/checkpoint.pth
```

This will:
- Load the meshes and their textures
- Compute vertex and normal maps from the meshes
- Run the HybridMQA model and output a quality score

You should raise `--vcmesh` flag if the input meshes are vertex-color meshes.

### Training & Generalization Test

Please refer to [`datasets/README.md`](datasets/README.md) for detailed instructions on dataset preparation and dataset directory structuring.

Before beginning the training/testing process, you need to prepare normal and vertex maps for all the meshes of the
dataset:
```bash
python3 scripts/prepare_normal_vertex_maps.py --root_dir /path/to/dataset/root/dir
```
This will:
- Recursively search the dataset directory for all `.obj` files.
- Generate vertex and normal maps for each mesh using UV mapping.
- Save the computed maps in NumPy format alongside the corresponding `.obj` file.

Then, to train HybridMQA run:

```bash
bash scripts/train_[dataset].sh /path/to/dataset/root/dir
```
We currently support training and testing on four datasets:
- YN2023 (Nehmé et al.): [Dataset Link](https://yananehme.github.io/publications/2022-ACM-TOG)
- TMQA (SJTU-TMQA): [Dataset Link](https://ccccby.github.io/)
- TSMD: [Dataset Link](https://multimedia.tencent.com/resources/tsmd)
- VCMesh (CMDM): [Dataset Link](https://yananehme.github.io/publications/2020-IEEE-TVCG)

You can implement support for additional datasets by creating a custom dataset 
class in [`hybridmqa/data/dataset.py`](hybridmqa/data/dataset.py) and start training/testing on them.

**Optionally**, to speed up training and testing, you can preprocess the dataset by iterating
through it once and saving mesh information as `.pt` tensors using the `save_tensor` argument
in the `load_objs_as_meshes` function 
(defined in [`hybridmqa/data/data_io_utils.py`](hybridmqa/data/data_io_utils.py)).
This will store mesh data (e.g., vertices, faces, normals) in a more efficient format.
During the actual training or testing process, you can use the `load_tensor` argument
to load these `.pt` files instead of `.obj` files, significantly reducing loading time.

Finally, to test generalization of HybridMQA on a dataset run:

```bash
bash  scripts/test.sh test_dataset_name /path/to/dataset/root/dir train_dataset_name /path/to/checkpoint
```
`train_dataset_name` argument is included only for logging purposes.

For example, to test the generalization of HybridMQA on the `TSMD` dataset using a
checkpoint trained on the `TMQA` dataset, run:

```bash
bash scripts/test.sh TSMD /path/to/TSMD/dataset TMQA /path/to/checkpoint.pth
```

---

## Repository Structure

```
HybridMQA/
│
├── scripts/              # Training and test scripts
├── datasets/             # datasets folder - containing README for dataset preparations
├── hybridmqa/            # Core HybridMQA model
    ├── data/             # dataset classes and data io utils
    ├── model/            # HybridMQA architecture classes
    ├── run/              # training, testing, and inference implementations
├── docs/                 # Project webpage
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{sarvestani2024hybridmqa,
    title={HybridMQA: Exploring Geometry-Texture Interactions for Colored Mesh Quality Assessment},
    author={Sarvestani, Armin Shafiee and Tang, Sheyang and Wang, Zhou},
    journal={arXiv preprint arXiv:2412.01986},
    year={2024}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions or collaborations, feel free to reach out via [email](mailto:a5shafie@uwaterloo.ca) or create an [issue](https://github.com/arshafiee/hybridmqa/issues).

---


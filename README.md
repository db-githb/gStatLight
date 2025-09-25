<h1 align="center">gStatLight</h1>

A command-line tool to **spotlight central objects** in 3D Gaussian Splatting (3DGS) reconstructions.  Given a walk-around video, gStatLight isolates the subject (e.g., a car) by keeping only Gaussians within a small radius of the scene’s origin.

<p align="center">
  <img src="README_images/before.png" alt="Original 3DGS Reconstruction" width="45%" />
  <img src="README_images/after.png" alt="After applying gSplatLight" width="45%" />
</p>

## 💾 Installation

### 1. Create & activate the Conda environment
```bash
conda create -n gsl python
conda activate gsl
```
### 2. Install dependencies

```bash
# Change directories to project root (gStatLight/):
cd <project-dir: gStatLight>

# Install the gStatLight package and its CLI entrypoints:
pip install .
```

## 📂 File Structure (Input Layout)

The tool requires the following structure:

```text
gCull/
├── data/
│   └── <experiment-name>/
│       ├── colmap/
│       ├── images/
│       └──  transforms.json
|
└── outputs/
    └── <experiment-name>/
        └── splatfacto/
            └── <model-name>/
                └── config.yml          ← point to this config file for `statlight`
                └── {...}_statlight.ply ← statlight output saved here
```

## 🚀 Execution

From your project root:

```
gsl statlight \
  --load-config <path/to/config.yml>
```

The final culled 3DGS model is saved alongside your ```config.yml``` as ```{experiment_name}_{model_name}_statlight.ply``` file.

## 🛠️ Acknowledgements

This work is built upon and heavily modifies the Nerfstudio/Splatfacto codebases.


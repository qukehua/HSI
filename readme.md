# InterFlow: Trajectory-Aware Flow Matching for Human Interaction Motion
Generation In Dynamic Scenes

## Prerequisites

To run the code, you need to have the following installed:

- Python 3.10
- Required Python packages (specified in `requirements.txt`)

## Installation

1. **Clone the Repository**:
    ```sh
    git clone git@github.com:mileret/lingo-release.git
    ```

2. **Download Checkpoints, Data, and SMPL-X Models**:
    - Download the necessary files and folders from [this link](https://drive.google.com/file/d/1L2V8RlPMAhWF93o_RpIznO_bacjSSLqu/view?usp=drive_link).
    - Extract `lingo_utils.zip`, and place the four files and folders (`dataset`, `ckpts`, `smpl_models`, `vis.blend`) at the root of the project directory.


3. **Install Python Packages**:
    ```sh
    conda create -n hsi python=3.10 -y
    conda activate hsi

    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Install Blender**:
    - We use [Blender](https://www.blender.org/) for visualization of the result.
    - Please download Blender3.6 from its [official website](https://download.blender.org/release/Blender3.6/).
    - (Optional) Then, download [SMPL-X Blender Add-on](https://smpl-x.is.tue.mpg.de/download.php) and activate it in Blender.

## Inference and Visualization

1. **Get Model Input**:

    Open `vis.blend` with [Blender](https://www.blender.org/). Change the `text`, `start_location`, `end_goal` and `hand_goal`. Then run `get_input` in `vis.blend`.

2. **Inference**:

    To synthesize human motions using our Flow Matching model, run

    ```sh
    cd code
    python sample.py
    ```

3. **Visualization in Blender**:

    Run `vis_output` in `vis.blend`.

    The generated human motion will be displayed in Blender.


# Training and Evaluation
## Overview

This README provides instructions on setting up and training the FlowHSI model with the LINGO/HSI, TRUMANS, or OMOMO datasets.

## Prerequisites

Before you begin, make sure you have the following software installed:

```sh
pip install -r requirements.txt
```

## Model Training

Navigate to the `code` directory:

```bash
cd code
```

### LINGO / HSI

Create the autoregressive window dataset:

```bash
python preprocess_window_dataset.py \
  --dataset-dir /share/qkh/dataset/lingo \
  --output-dir /share/qkh/dataset/lingo/window_t16_s3 \
  --window-size 16 \
  --step 3
```

Create scene-disjoint train/validation/test splits and mirror them to the
preprocessed window folder:

```bash
python create_dataset_splits.py \
  --dataset-dir /share/qkh/dataset/lingo \
  --window-dir /share/qkh/dataset/lingo/window_t16_s3
```

Train with the LINGO dataloader:

```bash
python train.py --config-name config_train_lingo \
  dataset.folder=/share/qkh/dataset/lingo/window_t16_s3 \
  dataset.scene_source_dir=/share/qkh/dataset/lingo
```

The LINGO setting uses `code/config/dataset/lingo.yaml`, with 28 joints and 768-D CLIP text features.

Train a LINGO motion evaluator for evaluator-space FID/Diversity:

```bash
python train_motion_evaluator.py --config-name config_motion_evaluator_lingo
```

### TRUMANS

Place the TRUMANS official files under `../dataset/trumans/trumans`, including:

```text
human_joints.npy
human_orient.npy
action_label.npy
idx_start.npy
scene_flag.npy
object_flag.npy
object_mat.npy
Object/
Scene/
```

Create scene-group-disjoint splits once. This avoids leakage between heavily
overlapping raw windows:

```bash
python create_dataset_splits.py \
  --dataset-dir ../dataset/trumans/trumans
```

Train with the TRUMANS dataloader:

```bash
python train.py --config-name config_train_trumans
```

If your TRUMANS folder is elsewhere, override the dataset path:

```bash
python train.py --config-name config_train_trumans \
  dataset.folder=/path/to/trumans/trumans
```

The TRUMANS setting uses `code/config/dataset/trumans.yaml`, with 24 joints, 10-D action labels, and object conditions from `object_flag.npy`, `object_mat.npy`, and `Object/*.npy`.

Train a TRUMANS motion evaluator for evaluator-space FID/Diversity:

```bash
python train_motion_evaluator.py --config-name config_motion_evaluator_trumans \
  dataset.folder=/path/to/trumans/trumans
```

### OMOMO

Place the official OMOMO files under `../dataset/OMOMO/data`. In particular,
the adapter consumes the official 120-frame train/test window files,
normalization statistics, and captured object meshes:

```text
train_diffusion_manip_window_120_cano_joints24.p
test_diffusion_manip_window_120_processed_joints24.p
min_max_mean_std_data_window_120_cano_joints24.p
captured_objects/
```

Train on the official subject-based training set:

```bash
python train.py --config-name config_train_omomo
```

OMOMO already separates subjects 1-15 for training and subjects 16-17 for
testing, so `create_dataset_splits.py` detects OMOMO and skips split generation.
The default training config keeps the official test subjects unloaded and does
not use them for validation. To explicitly monitor the official test set, add
`use_validation=true dataset.load_test=true val_split=test`.

The training script instantiates the dataloader from the selected dataset config, builds the conditional Flow Matching sampler from `./code/config/sampler/pelvis.yaml`, and trains the model using the configurations in `./code/config`.

## Model Evaluation


#FID, Diversity, Multi-modality, Precision
```sh
python evaluation.py ^
  --generated ..\results\outputs\model.pkl ^
  --reference gt ^
  --metrics interactive ^
  --smpl-dir ..\smpl_models
```

Use a trained motion evaluator checkpoint to compute FID/Diversity/Precision in evaluator embedding space instead of the fallback hand-crafted joint features:

```sh
python evaluation.py ^
  --generated ..\results\outputs\*.pkl ^
  --reference-dataset ..\dataset\lingo ^
  --reference-joints-ind 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,34,40,49 ^
  --motion-evaluator-checkpoint ..\results\motion_evaluator_lingo\checkpoints\best_model.pth ^
  --metrics interactive ^
  --smpl-dir ..\smpl_models
```

For TRUMANS, use the TRUMANS evaluator checkpoint and the TRUMANS reference files/split. The evaluator checkpoint must be trained with the same joint set as the motions being evaluated.

#Human-Object Contact Precision, Recall, F1
```sh
python evaluation.py ^
  --generated ..\results\outputs\*.pkl ^
  --reference ..\dataset\OMOMO\data\test_diffusion_manip_seq_joints24.p ^
  --metrics reaching ^
  --contact-threshold 0.05
```

The `reaching` metric group reports OMOMO-style `contact_precision`, `contact_recall`, and `contact_f1_score` when generated/reference samples contain object vertices such as `obj_verts`, `object_verts`, or `object_points`.

#Pene%, Pene mean, Pene max, Foot Sliding
```sh
python evaluation.py ^
  --generated ..\results\outputs\model.pkl ^
  --metrics locomotion ^
  --smpl-dir ..\smpl_models ^
  --scene-occ ..\dataset\Scene_vis ^
  --compute-vertices ^
  --body-points vertices
```

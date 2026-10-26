# InterFlow: Trajectory-Aware Flow Matching for Human Interaction Motion Generation In Dynamic Scenes

## Prerequisites

To run the code, you need to have the following installed:

1. **Install Python Packages**:
    ```sh
    conda create -n hsi python=3.10 -y
    conda activate hsi

    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```


2. **Download SMPL-X Models**:
    [SMPL-X]( https://drive.google.com/file/d/1BLUWs_jpNZs_D2FiCEaBoyy4V2wIVl_u/view?usp=drive_link)



3. **Install Blender**:
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
  --dataset-dir ../../datasets/lingo \
  --output-dir ../../datasets/lingo/window_t16_s3 \
  --window-size 16 \
  --step 3
```

Create scene-disjoint train/validation/test splits and mirror them to the
preprocessed window folder:

```bash
python create_dataset_splits.py \
  --dataset-dir ../../datasets/lingo \
  --window-dir ../../datasets/lingo/window_t16_s3
```

Train with the LINGO dataloader:

```bash
python train.py --config-name config_train_lingo \
  dataset.folder=../../datasets/lingo/window_t16_s3 \
  dataset.scene_source_dir=../../datasets/lingo
```

The LINGO setting uses `code/config/dataset/lingo.yaml`, with 28 joints and 768-D CLIP text features.

Train a LINGO motion evaluator for evaluator-space FID/Diversity:

```bash
python train_motion_evaluator.py --config-name config_motion_evaluator_lingo
```

### TRUMANS

Place the TRUMANS official files under `../../datasets/trumans/trumans`, including:

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
  --dataset-dir ../../datasets/trumans/trumans
```

Train with the TRUMANS dataloader:

```bash
python train.py --config-name config_train_trumans
```

If your TRUMANS folder is elsewhere, override the dataset path:

```bash
python train.py --config-name config_train_trumans \
  dataset.folder=../../datasets/trumans/trumans
```

The TRUMANS setting uses `code/config/dataset/trumans.yaml`, with 24 joints, 10-D action labels, and object conditions from `object_flag.npy`, `object_mat.npy`, and `Object/*.npy`.

Train a TRUMANS motion evaluator for evaluator-space FID/Diversity:

```bash
python train_motion_evaluator.py --config-name config_motion_evaluator_trumans \
  dataset.folder=../../datasets/trumans/trumans
```

### OMOMO

Place the official OMOMO files under `../../datasets/data`. In particular,
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

### Hybrid BPS object geometry

TRUMANS and OMOMO encode object shape with 256 deterministic basis points in
the object-local metric frame. The global object token is computed from vector
BPS residuals. The same residuals reconstruct surface proxy points, which are
placed at each frame using the 9-D object motion condition (3-D translation and
6-D rotation). Human-object cross queries operate on these dynamic proxies.
The original sampled point cloud remains available as a
fallback for data without BPS fields. Object-conditioned checkpoints trained
before this change do not contain the BPS encoder parameters and should be
retrained. The basis size, radius, and seed are configured by
`bps_num_points`, `bps_radius`, and `bps_seed` in the dataset configuration.

### Signed object penetration supervision

The training collision term now queries object SDFs instead of penalizing
unsigned proximity to surface proxies. With negative SDF values inside an
object, it minimizes `mean(relu(-sdf_meters - tolerance_m)^2)` over selected
joints and valid frames with SDF supervision. The default
`human_object_penetration_tolerance_m: 0.0` penalizes every negative distance;
zero-distance contact and exterior points receive no penalty. A positive
tolerance permits that many meters of penetration, not an exterior clearance.
This is a joint-level soft constraint, not a whole-body collision guarantee.

Before querying, predicted joints are denormalized using the dataset's axis
ranges. The inverse per-frame mesh translation, rotation, and object scale
maps them into the SDF rest frame; queried distances are multiplied by the
object scale to recover meters. OMOMO's SDF xyz layout, metadata, and sign
follow its [official evaluation implementation](https://github.com/lijiaman/omomo_release/blob/main/evaluation_metrics.py).
Queries use the original grid resolution with differentiable trilinear
interpolation. Grid discretization still limits accuracy near the surface;
points outside the finite grid are not extrapolated as interior.

OMOMO training enables `human_object_collision_mode: sdf` and requires
`rest_object_sdf_256_npy_files/*.obj.npy` with corresponding `.obj.json`
metadata under the dataset folder. Each grid is cached by object/device, not
duplicated across frames or batch items. The dataset sends mesh-origin poses
from `obj_trans`, `obj_rot_mat`, and `obj_scale`, rather than COM trajectories.

The released mop/vacuum sequences have independently moving top and bottom
parts, but the supplied SDFs describe whole rest objects. The default explicit
`dataset.object_sdf_excluded_objects: [mop, vacuum]` therefore excludes them
and emits a warning. To supervise them, supply matching
`<object>_cleaned_simplified_top.obj.npy/.json` and
`<object>_cleaned_simplified_bottom.obj.npy/.json`, then remove the exclusions.
The loader and loss support the union of these separately posed part SDFs.
Missing non-excluded geometry raises an error; there is no convex-hull or
unsigned-distance fallback. `human_object_collision_coverage` logs the fraction
of valid object frames covered by SDF supervision in each batch. Excluded
frames do not dilute the mean penetration loss.

TRUMANS and LINGO currently disable this term (`mode: disabled`, weight `0`):
TRUMANS lacks accurate SDF inputs and LINGO supplies no object geometry.
Enabling a nonzero SDF loss without a dataset SDF adapter raises an error.
The former normalized collision margin has been renamed
`human_object_query_margin`; it remains only a cross-query feature, separate
from the penetration tolerance in meters. No learned parameter shapes changed,
but existing checkpoints and reported metrics do not acquire the benefit of
this new objective without training/fine-tuning and fresh evaluation.

## Model Evaluation


#FID, Diversity, Multi-modality, Precision
```sh
python evaluation.py \
  --generated ../results/outputs/model.pkl \
  --reference gt \
  --metrics interactive \
  --smpl-dir ../smpl_models
```

Use a trained motion evaluator checkpoint to compute FID/Diversity/Precision in evaluator embedding space instead of the fallback hand-crafted joint features:

```sh
python evaluation.py \
  --generated ../results/outputs/*.pkl \
  --reference-dataset ../../datasets/lingo \
  --reference-joints-ind 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,34,40,49 \
  --motion-evaluator-checkpoint ../results/motion_evaluator_lingo/checkpoints/best_model.pth \
  --metrics interactive \
  --smpl-dir ../smpl_models
```

For TRUMANS, use the TRUMANS evaluator checkpoint and the TRUMANS reference files/split. The evaluator checkpoint must be trained with the same joint set as the motions being evaluated.

#Human-Object Contact Precision, Recall, F1
```sh
python evaluation.py \
  --generated ../results/outputs/*.pkl \
  --reference ../../datasets/data/test_diffusion_manip_seq_joints24.p \
  --metrics reaching \
  --contact-threshold 0.05
```

The `reaching` metric group reports OMOMO-style `contact_precision`, `contact_recall`, and `contact_f1_score` when generated/reference samples contain object vertices such as `obj_verts`, `object_verts`, or `object_points`.

#Pene%, Pene mean, Pene max, Foot Sliding
```sh
python evaluation.py \
  --generated ../results/outputs/model.pkl \
  --metrics locomotion \
  --smpl-dir ../smpl_models \
  --scene-occ ../../datasets/lingo/Scene_vis \
  --compute-vertices \
  --body-points vertices
```

# FlowHSI: Goal-Conditioned Flow Matching for Human-Scene Interaction Motion Synthesis

This project is positioned around **Flow Matching**, not a general-purpose world model. The model learns a conditional velocity field that transports noisy motion windows into goal-satisfying human-scene interaction motions under scene, text, pelvis-goal, hand-goal, and progress conditions.

## Pipeline

1. **Windowed data construction**: `preprocess_window_dataset.py` converts LINGO/HSI sequences into fixed-size autoregressive windows with normalized human joints, valid-frame masks, text CLIP features, scene IDs, pelvis goals, hand goals, progress indicators, and terminal/completion labels.
2. **Goal-conditioned Flow Matching training**: `train.py` samples a noisy interpolation time, builds the masked window condition, and trains the sampler/model to regress the velocity field from the noisy window toward the target motion.
3. **Autoregressive ODE sampling**: `sample.py` starts each window from noise, keeps prefix frames fixed, and integrates the learned field with deterministic Flow Matching ODE steps. The default sampler uses 50 steps and can be ablated with fewer steps.
4. **Scene-aware and goal-aware control**: language, occupancy grids, pelvis goals, hand goals, and progress tokens are injected as conditioning tokens. Locomotion segments can use A* path guidance, while future collision/contact/arrival guidance can be inserted as ODE correction terms.
5. **Completion-aware rollout**: the completion head predicts whether the current window finishes the requested segment, making autoregressive stitching more stable than relying only on a noisy reverse diffusion trajectory.
6. **SMPL-X conversion and evaluation**: generated joints are converted to SMPL-X for visualization, then evaluated with interaction, locomotion, reaching, penetration, and diversity-style metrics.

## Why Flow Matching Fits This Task

- **Faster window sampling**: autoregressive rollout samples one window at a time, so replacing a long DDPM reverse chain with Flow Matching ODE integration directly reduces per-window latency.
- **Less stochastic jitter**: DDPM injects random noise at every reverse step, which can create visible shaking when windows are stitched. Flow Matching uses a deterministic velocity field during sampling, making autoregressive transitions smoother.
- **More direct goal conditioning**: this task is not unconstrained random motion generation. Motions must satisfy scene, text, pelvis-goal, hand-goal, and progress conditions, and Flow Matching naturally learns the velocity from noise toward the conditioned target motion.
- **Guidance-friendly sampling**: deterministic ODE updates make it straightforward to add collision, contact, or object-arrival correction terms during integration.
- **More stable completion prediction**: the completion head is trained alongside the same smooth motion field, so segment stopping is less exposed to random reverse-process perturbations.

## Related Flow Matching Work

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) introduced simulation-free CNF training by regressing vector fields over fixed probability paths, with efficient ODE sampling.
- [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482) developed conditional Flow Matching and OT-CFM, emphasizing stable regression objectives and deterministic flow-model inference.
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) showed that straight transport paths can enable coarse, fast ODE sampling.
- [Motion Flow Matching for Human Motion Synthesis and Editing](https://arxiv.org/abs/2312.08895) applied Flow Matching to human motion, reducing diffusion-style sampling complexity to very few steps while supporting editing through ODE trajectory rewriting.
- [FlowMotion: Target-Predictive Flow Matching for Realistic Text-Driven Human Motion Generation](https://arxiv.org/abs/2504.01338) used conditional Flow Matching for text-driven human motion and emphasized reduced jitter, stability, realism, and computational efficiency.
- [HY-Motion 1.0](https://arxiv.org/abs/2512.23464) scaled DiT-based Flow Matching for text-to-motion generation, showing the direction is viable for large motion-generation models.
- [Riemannian Motion Generation](https://arxiv.org/abs/2603.15016) extended the idea to Riemannian Flow Matching for geometry-aware human motion representations.

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

This README provides instructions on setting up and training the FlowHSI model with either the LINGO/HSI window dataset or the TRUMANS raw dataset.

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

Train with the LINGO dataloader:

```bash
python train.py --config-name config_train \
  dataset=lingo \
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

The training script instantiates the dataloader from the selected dataset config, builds the conditional Flow Matching sampler from `./code/config/sampler/pelvis.yaml`, and trains the model using the configurations in `./code/config`.

## Model Evaluation


#FID, Diversity, Multi-modality, Precision, Recall, F1
```sh
python evaluation.py ^
  --generated ..\results\outputs\model.pkl ^
  --reference gt ^
  --metrics interactive ^
  --smpl-dir ..\smpl_models
```

Use a trained motion evaluator checkpoint to compute FID/Diversity/Precision/Recall/F1 in evaluator embedding space instead of the fallback hand-crafted joint features:

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


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

    To synthesis human motions using our model, run

    ```sh
    cd code
    python sample_lingo.py
    ```

3. **Visualization in Blender**:

    Run `vis_output` in `vis.blend`.

    The generated human motion will be displayed in Blender.


# Training and Evaluation
## Overview

This README provides instructions on setting up and training our model using the LINGO dataset.

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

To start training the model, run the training script from the command line:

```bash
python train_lingo.py
```

The training script will automatically load the dataset, set up the model, and commence training sessions using the configurations in `./code/config` folder.

## Model Evaluation


#FID, Diversity, Multi-modality, Precision, Recall, F1
```sh
python evaluation.py ^
  --generated ..\results\outputs\model.pkl ^
  --reference gt ^
  --metrics interactive ^
  --smpl-dir ..\smpl_models
```

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

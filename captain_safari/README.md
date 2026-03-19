# Captain Safari: Model Training and Inference

<p align="center">
    <img src="../assets/pipeline.png"/> <br />
    <em> 
    **Method overview.** Captain Safari builds a local world memory and, given a query camera pose, retrieves pose-aligned tokens that summarize the scene. These tokens then condition video generation along the user-specified trajectory, preserving a stable 3D layout.
    </em>
</p>

## 1. Overview

We open source the full training and inference code for Captain Safari, built upon [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (commit `db124fa`). Our codebase enables pose-conditioned memory retriever warmup, efficient DiT joint training with pre-computed latents, and world consistent video generation.

To implement the above process, we structure our pipeline into the following sequential steps:

- [**Sec. 3. Memory Retriever Warmup**](#3-memory-retriever-warmup): Scripts to pre-train the pose-conditioned memory retriever to predict target 3D world memory.
- [**Sec. 4. Pre-tokenization**](#4-optional-pre-tokenization): Optional but highly recommended scripts to pre-encode text, video, and image latents to significantly accelerate training and reduce VRAM usage.
- [**Sec. 5. Training**](#5-training): Distributed LoRA training scripts that jointly optimize the DiT model with the memory retriever, featuring our core modifications to the DiffSynth-Studio framework.
- [**Sec. 6. Inference**](#6-inference): Python scripts for evaluating the trained models, downloading pre-trained checkpoints, and generating 3D-consistent videos under complex camera trajectories.
 

## 2. Installation

```bash
conda create -n captain_safari python=3.10 -y && conda activate captain_safari
cd captain_safari && pip install -e .

ln -s ../opensafari ./data
```

## 3. Memory Retriever Warmup

As described in **Section 3.2** and **5.1** of our paper, before jointly training the retriever with the DiT model, we first warm up the pose-conditioned memory retriever. This step trains the retriever to accurately predict the target 3D world memory ($m_{t_1}$) at a given query camera pose ($p_{t_1}$) by attending to the local memory context ($\mathcal{M}_{local}$).

We provide a distributed training script [`retriever_wamup.py`](retriever_wamup.py) to run this warm-up phase, optimizing a MSE loss.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="localhost" \
  --master_port=12345 \
  retriever_wamup.py \
  --data_dir ./data/ \
  --train_metadata metadata.csv \
  --steps 5000 \
  --batch_size 4 \
  --rope_mode 3d
```

This script will:
1. **Load Preprocessed Data**: Reads the `metadata.csv` generated from the `OpenSafari` data curation pipeline, which includes the camera trajectories and StreamVGGT memory.
2. **Train the Retriever**: Optimizes the Memory Encoder and Query Encoder using 3D-aware Rotary Position Encoding (3D RoPE) to aggregate pose-aligned memory tokens.
3. **Save Checkpoints**: Periodically saves the model weights to the `checkpoints/` directory to serve as initialization for the downstream joint training.

### Key Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data_dir` | `./data/` | Base directory containing the preprocessed data and CSV files. |
| `--train_metadata` | `metadata.csv` | Path to the metadata index file relative to `--data_dir`. |
| `--steps` | `5000` | Total number of training steps. |
| `--batch_size` | `4` | Batch size per GPU. |
| `--rope_mode` | `3d` | RoPE mode to use for spatial-temporal encoding (`3d`, `1d`, or `none`). |
| `--use_bf16` | `False` | Use BFloat16 precision for memory-efficient training. |

## 4. (Optional) Pre-tokenization

To significantly accelerate the training process and reduce GPU memory consumption (~13GB VRAM saved by skipping Text Encoder and VAE loading), we highly recommend pre-encoding the dataset. This step will pre-compute and store the prompt embeddings, video latents, and image latents to disk.

```bash
# Step 1: Pre-encode prompt embeddings and video latents
# Outputs: prompt_embeddings/*.pt, video_latents/*.pt, and metadata.encoded.csv
bash ./utils/run_pretokenize_multi_gpu.sh

# Step 2: Pre-encode image latents (required for camera control training)
# Outputs: image_y_latents/*.pt, image_single_latents/*.pt, and metadata.encoded_with_image.csv
bash ./utils/run_pretokenize_image_multi_gpu.sh

# Step 3: Merge the generated CSV files
# Outputs: A final metadata.encoded.csv containing all pre-encoded latent paths
python ./utils/merge_csv.py \
  --base_csv data/metadata.encoded.csv \
  --source_csv data/metadata.encoded_with_image.csv \
  --columns image_y_latent,image_single_latent \
  --output_csv data/metadata.encoded.csv
```

These scripts will process the dataset in parallel across multiple GPUs and generate `.pt` files containing the pre-computed tensors. A new unified metadata CSV will be created to link these files. 

After pre-tokenization, you can run the memory-efficient training script by specifying the `--use_preencoded` flag in [`Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc.sh`](./examples/wanvideo/model_training/lora/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc.sh) and skip the encoding process to accelerate inference in [`Wan2.2-Fun-5B-Control-Camera_Captain-Safari.py`](./examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.py)

## 5. Training

We provide several bash scripts to launch the LoRA training for the DiT model.

- **Train Captain-Safari**
  This script trains the model with the proposed pose-conditioned memory retriever, optimizing the DiT to leverage the 3D-aware world memory.
```bash
bash ./examples/wanvideo/model_training/lora/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.sh
```

- **Train Captain-Safari with Pre-encoded inputs** (Highly Recommended)
  Same as above, but uses the pre-computed latents and embeddings to reduce memory usage and accelerate training.
```bash
bash ./examples/wanvideo/model_training/lora/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc.sh
```

### Codebase Development

Our implementation is built upon [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (commit [db124fa](https://github.com/modelscope/DiffSynth-Studio/commit/db124fa6bc63dda4401b025228c7e254b804a29f)). To support 3D world memory and pre-tokenization, we made several core modifications to the original framework:

- **[`examples/wanvideo/model_training/train.py`](./examples/wanvideo/model_training/train.py)**: Modified the `forward_preprocess` and data loading logic to support bypassing Text/VAE encoding when reading pre-encoded `.pt` files. Additionally, we unfreezed the `memory_emb` and `memory_retriever` parameters for memory joint training.
- **[`diffsynth/trainers/utils.py`](./diffsynth/trainers/utils.py)**: Updated the `VideoDataset` class to parse complex metadata (e.g., StreamVGGT memory, 6-DoF camera trajectories) and conditionally load pre-encoded image/video latents. Integrated Wandb logging and dynamic gradient clipping.
- **[`diffsynth/pipelines/wan_video_new.py`](./diffsynth/pipelines/wan_video_new.py)**: Adjusted the pipeline units (e.g., `WanVideoUnit_PromptEmbedder`, `WanVideoUnit_InputVideoEmbedder`) to seamlessly accept pre-computed tensors directly. Implemented `load_memory` and `load_cross_attn` to load 3D world memory checkpoint weights.
- **[`diffsynth/models/wan_video_dit.py`](./diffsynth/models/wan_video_dit.py)**: **(Core Contribution)** Implemented the *Memory-conditioned DiT* architecture (see **Section 3.2** of the paper). Added `memory_retriever` (Query Encoder + Cross Attention), `memory_emb`, and injected the retrieved 3D world tokens into the DiT layers via dedicated `memory_cross_attn`.
- **[`diffsynth/models/wan_video_vae.py`](./diffsynth/models/wan_video_vae.py)**: Patched the VAE encoder to return pure dictionaries when requested and bypass tiled encoding constraints, facilitating the export of standalone image/mask latents during pre-tokenization.
- **[`diffsynth/models/model_manager.py`](./diffsynth/models/model_manager.py)**: Added utility functions (`_init_memory_emb`, `_init_memory_retriever`) to properly initialize and format the newly introduced memory cross-attention and memory retriever parameters alongside LoRA weights.
- **[`diffsynth/models/wan_video_camera_controller.py`](./diffsynth/models/wan_video_camera_controller.py)**: Enhanced the camera pose processor with `convert_to_local_coordinates` and `generate_plucker_embedding_cuda` to calculate Plücker embeddings dynamically for complex multi-view generation.


## 6. Inference

We provide Python scripts to run inference with the trained models.

### Preparation: Download Demo Data and Checkpoints

We host the demo dataset and the pre-trained checkpoints on Hugging Face. Please download and extract them before running inference:

```bash
# 1. Download checkpoint to the correct directory
huggingface-cli download Johnson111788/Captain-Safari \
  --include "models/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc/*" \
  --local-dir ./

# 2. Download and extract demo data for OpenSafari
huggingface-cli download Johnson111788/Captain-Safari demo_data.tar.gz --local-dir ./
tar -xzvf demo_data.tar.gz -C ./data
```

### Infer Captain-Safari

```bash
python ./examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.py
```

### Infer Base Model (Wan2.2-Fun-5B-Control-Camera)

```bash
python ./examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-5B-Control-Camera.py
```

<p align="center">
    <img src="../assets/prediction.gif"/> <br />
    <em>
    **Inference Result**. The above videos show the provided demo sample generated by Captain Safari. You should expect high-quality, geometry-consistent video outputs from the model, demonstrating accurate 3D camera motion and scene awareness.
    </em>
</p>

### Preliminary Experiments

We also include scripts from our early experiments based on the base Image-to-Video models.

- **Infer Wan2.2-TI2V-5B**
```bash
python ./examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B.py
```

- **Autoregressive Generation (Theoretically Infinite Length)**
  Experimental scripts for autoregressive video generation, optionally conditioned on StreamVGGT outputs.
```bash
# Standard autoregressive generation
python ./examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B-AR.py

# Autoregressive generation with StreamVGGT conditioning
python ./examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B-StreamVGGT-AR.py
```

## 7. Citation

If you find this repository helpful, please consider citing:

```
@article{chou2025captain,
  title={Captain Safari: World Engine with Pose-Aligned 3D Memory},
  author={Chou, Yu-Cheng and Wang, Xingrui and Li, Yitong and Wang, Jiahao and Liu, Hanting and Xie, Cihang and Yuille, Alan and Xiao, Junfei},
  journal={arXiv preprint arXiv:2511.22815},
  year={2025}
}
```
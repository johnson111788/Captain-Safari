import os
import pandas as pd
import torch
import numpy as np
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import torch.multiprocessing as mp

def process_videos_on_gpu(gpu_id, data_chunk, model_path, data_path, save_path, use_preencoded=False):
    """
    this function will run on a single specified GPU, processing the video data assigned to it.
    Support using pre-encoded prompt embedding and video latent.
    """
    # 1. Set the GPU to be used for this process
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Process started, will run on this device: {device}")
    print(f"[GPU {gpu_id}] Using pre-encoded data: {use_preencoded}")

    # 2. Load the model independently into the specified GPU
    #    Each process needs its own model instance
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.2-Fun-5B-Control-Camera", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.2-Fun-5B-Control-Camera", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.2-Fun-5B-Control-Camera", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ],
    )
    
    pipe.dit.use_memory_retrieval = True  # Use memory retrieval
    pipe.dit.use_memory_cross_attn = True  # Use memory cross attention
    pipe.load_lora(pipe.dit, model_path, alpha=1)
    pipe.load_memory(pipe.dit, model_path)
    pipe.load_cross_attn(pipe.dit, model_path)  # Load memory cross-attention weights
    pipe.load_memory_retriever_from_dit(pipe.dit, model_path)  # Load memory retriever weights
    pipe.enable_vram_management()
    
    # Ensure memory_retriever is on the correct device
    if hasattr(pipe.dit, 'memory_retriever'):
        pipe.dit.memory_retriever.to(device=f'cuda:{gpu_id}', dtype=torch.bfloat16)
    print(f"[GPU {gpu_id}] Model loaded.")


    # 3. Process the data assigned to this GPU
    for _, row in data_chunk.iterrows():
        video_name = row['video'].split('/')[-1]
        
        # Check if using pre-encoded data
        if use_preencoded and 'prompt_embedding' in row and 'video_latent' in row:
            # Use pre-encoded data
            prompt_embedding_path = row['prompt_embedding']
            video_latent_path = row['video_latent']
            
            # Load pre-encoded prompt embedding and video latent
            prompt_embedding = torch.load(os.path.join(data_path, prompt_embedding_path), map_location='cpu')
            video_latent = torch.load(os.path.join(data_path, video_latent_path), map_location='cpu')
            
            prompt = f"[Pre-encoded] {row.get('prompt', 'N/A')}"  # For display
            
            # Load pre-encoded input_image latent (for camera control)
            if 'image_y_latent' in row and 'image_single_latent' in row:
                image_y_path = row['image_y_latent']
                image_single_path = row['image_single_latent']
                image_y = torch.load(os.path.join(data_path, image_y_path), map_location='cpu')
                image_single = torch.load(os.path.join(data_path, image_single_path), map_location='cpu')
                input_image = {'y': image_y, 'single': image_single}
                print(f'[GPU {gpu_id}] Using pre-encoded data: prompt_emb {prompt_embedding.shape}, video_latent {video_latent.shape}, image_y {image_y.shape}')
            else:
                # No pre-encoded input_image, need to load original image
                input_image = VideoData(f"{data_path}/videos/{video_name}", height=704, width=1280)[0]
                print(f'[GPU {gpu_id}] Using pre-encoded data (no image latent): prompt_emb {prompt_embedding.shape}, video_latent {video_latent.shape}')
        else:
            # Use original data
            prompt = row['prompt']
            prompt_embedding = None
            video_latent = None
            input_image = VideoData(f"{data_path}/videos/{video_name}", height=704, width=1280)[0]
        
        # Load memory
        memory = row['memory']
        memory = np.load(os.path.join(data_path, memory), allow_pickle=True)
        # For VFMV, memory should be the first 4 frames of memory, shape: [4, 4, 782, 1024]
        memory = torch.tensor(memory, dtype=torch.bfloat16).unsqueeze(0)  # [1, 4, 4*782, 1024]
        
        # Load new VFMV camera parameters
        intrinsic_query_path = row['intrinsic_query']
        extrinsic_query_path = row['extrinsic_query'] 
        intrinsic_key_path = row['intrinsic_key']
        extrinsic_key_path = row['extrinsic_key']
        intrinsic_clip_path = row['intrinsic_clip']
        extrinsic_clip_path = row['extrinsic_clip']
        
        intrinsic_query = np.load(os.path.join(data_path, intrinsic_query_path))  # (1, 3, 3)
        extrinsic_query = np.load(os.path.join(data_path, extrinsic_query_path))  # (1, 3, 4)
        intrinsic_key = np.load(os.path.join(data_path, intrinsic_key_path))     # (4, 1, 3, 3) 
        extrinsic_key = np.load(os.path.join(data_path, extrinsic_key_path))     # (4, 1, 3, 4)
        intrinsic_clip = np.load(os.path.join(data_path, intrinsic_clip_path))
        extrinsic_clip = np.load(os.path.join(data_path, extrinsic_clip_path))

        # Convert to tensor and add batch dimension
        intrinsic_query = torch.tensor(intrinsic_query, dtype=torch.bfloat16).unsqueeze(0)  # [1, 1, 3, 3]
        extrinsic_query = torch.tensor(extrinsic_query, dtype=torch.bfloat16).unsqueeze(0)  # [1, 1, 3, 4]
        intrinsic_key = torch.tensor(intrinsic_key, dtype=torch.bfloat16).unsqueeze(0)      # [1, 4, 1, 3, 3]
        extrinsic_key = torch.tensor(extrinsic_key, dtype=torch.bfloat16).unsqueeze(0)      # [1, 4, 1, 3, 4]
        intrinsic_clip = torch.tensor(intrinsic_clip, dtype=torch.bfloat16).unsqueeze(0)  # [1, 20, 3, 3]
        extrinsic_clip = torch.tensor(extrinsic_clip, dtype=torch.bfloat16).unsqueeze(0)  # [1, 20, 3, 4]
        
        print(f'[GPU {gpu_id}] Start processing video: {video_name} | Memory: {memory.shape} | Prompt: {prompt}\n')
        # Prepare inference parameters
        inference_kwargs = {
            "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "memory": memory,  # The first 4 frames of memory [1, 4, 4*782, 1024]
            
            "intrinsic_query": intrinsic_query,  # Target frame intrinsic matrix [1, 1, 3, 3]
            "extrinsic_query": extrinsic_query,  # Target frame extrinsic matrix [1, 1, 3, 4]
            "intrinsic_key": intrinsic_key,      # The first 4 frames of intrinsic matrix [1, 4, 1, 3, 3]
            "extrinsic_key": extrinsic_key,      # The first 4 frames of extrinsic matrix [1, 4, 1, 3, 4]
            "extrinsic_clip": extrinsic_clip,    # Extrinsic matrix [1, 20, 3, 4]
            "intrinsic_clip": intrinsic_clip,    # Intrinsic matrix [1, 20, 3, 3]
            "height": 704,
            "width": 1280,
            "num_frames": 121,
            "seed": 1,
            "tiled": True,
        }
        
        # Add different parameters based on whether using pre-encoded data
        if use_preencoded and prompt_embedding is not None and video_latent is not None:
            # Use pre-encoded data for inference
            # Note: Here we need special handling, directly pass the pre-encoded tensor
            # We need to simulate the behavior of the pipeline, skip the encoding step
            inference_kwargs["prompt"] = prompt_embedding  # Directly pass the embedding
            inference_kwargs["input_video"] = video_latent  # Direct
            inference_kwargs["input_image"] = input_image
        else:
            # Use original data for inference
            inference_kwargs["prompt"] = prompt
            inference_kwargs["input_image"] = input_image

        # Use memory retrieval: the first 4 frames of memory + target pose -> generate the 8th frame
        video = pipe(**inference_kwargs)
        
        output_file = f"{save_path}/{os.path.splitext(video_name)[0]}_generated.mp4"
        save_video(video, output_file, fps=24, quality=10)
        print(f'[GPU {gpu_id}] Successfully saved video: {output_file}')

def main():
    # --- Global settings ---
    data_path = './data/'
    model_path = './models/train/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc/epoch-4.safetensors'

    epoch = model_path.split('/')[4].split('.')[0]
    model_name = model_path.split('/')[3]
    data_name = data_path.split('/')[1]

    # --- GPU detection and data splitting ---
    if not torch.cuda.is_available():
        print("Error: No available CUDA GPU.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} available GPUs.")

    data = pd.read_csv(f'{data_path}/metadata.csv')
    print(f"Total {len(data)} videos to process.")
    
    # Check if using pre-encoded data
    use_preencoded = 'prompt_embedding' in data.columns and 'video_latent' in data.columns and 'image_y_latent' in data.columns and 'image_single_latent' in data.columns
    if use_preencoded:
        print(f"✓ Detected pre-encoded data set! Will use pre-encoded prompt embeddings and video latents and image y latents and image single latents")
        print(f"  - Will skip Text Encoder and VAE loading, save memory and time")
        # Add "preencoded" tag in the save path
        save_path = f'./output/{data_name}/{model_name}.PreEnc-{epoch}'
    else:
        print(f"ℹ Use original data set, will perform complete encoding process")
        save_path = f'./output/{data_name}/{model_name}-{epoch}'
    
    os.makedirs(save_path, exist_ok=True)
    print(f"Output path: {save_path}\n")

    # Implement resume function: if the save_path already has the corresponding file, skip these already processed videos
    # Get the already generated file names (remove the _generated.png part)
    existing_files = set()
    for fname in os.listdir(save_path):
        if fname.endswith('_generated.mp4'):
            existing_files.add(fname.replace('_generated.mp4', ''))

    # Filter out already processed videos
    def is_not_processed(row):
        video_name = os.path.splitext(os.path.basename(row['video']))[0]
        return video_name not in existing_files

    data = data[data.apply(is_not_processed, axis=1)].reset_index(drop=True)
    print(f"Resume: skip existing videos, remaining {len(data)} videos to process.")

    # Use numpy.array_split to split the data frame into N parts (N = GPU count)
    # This method can handle the case where the data cannot be divided evenly by the number of GPUs
    data_chunks = np.array_split(data, num_gpus)

    # --- Start parallel processes ---
    # Set the method to start multiple processes to 'spawn', which is the most stable and recommended for CUDA
    mp.set_start_method('spawn', force=True)

    processes = []
    for gpu_id in range(num_gpus):
        # If a GPU is not assigned to data (for example, the number of clips is less than the number of GPUs), skip
        if len(data_chunks[gpu_id]) == 0:
            continue
            
        p = mp.Process(target=process_videos_on_gpu, args=(gpu_id, data_chunks[gpu_id], model_path, data_path, save_path, use_preencoded))
        p.start()
        processes.append(p)

    # Wait for all sub-processes to complete
    for p in processes:
        p.join()

    print("All videos have been processed.")


if __name__ == '__main__':
    main()
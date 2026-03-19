import os
import pandas as pd
import torch
import numpy as np
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import torch.multiprocessing as mp

def process_videos_on_gpu(gpu_id, data_chunk, data_path, save_path):
    """
    This function will run on a single specified GPU, processing the video data assigned to it.
    """
    # 1. Set the GPU to be used for this process
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Process started, will run on this device: {device}")

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
    pipe.enable_vram_management()

    # 3. Iterate through the video data assigned to this GPU and process it
    for _, row in data_chunk.iterrows():
        video_name = row['video'].split('/')[-1]
        prompt = row['prompt']
        
        intrinsic_clip_path = row['intrinsic_clip']
        extrinsic_clip_path = row['extrinsic_clip']
        
        intrinsic_clip = np.load(os.path.join(data_path, intrinsic_clip_path))
        extrinsic_clip = np.load(os.path.join(data_path, extrinsic_clip_path))

        intrinsic_clip = torch.tensor(intrinsic_clip, dtype=torch.bfloat16).unsqueeze(0)  # [1, 20, 3, 3]
        extrinsic_clip = torch.tensor(extrinsic_clip, dtype=torch.bfloat16).unsqueeze(0)  # [1, 20, 3, 4]

        print(f'[GPU {gpu_id}] Start processing video: {video_name} | Prompt: {prompt}\n')
        
        try:
            input_image = VideoData(f"{data_path}/videos/{video_name}", height=704, width=1280)[0]
            video = pipe(
                prompt=prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                input_image=input_image,
                height=704, width=1280,
                extrinsic_clip=extrinsic_clip,
                intrinsic_clip=intrinsic_clip,
                num_frames=121,
                seed=1, tiled=True,
            )
            
            output_file = f"{save_path}/{os.path.splitext(video_name)[0]}_generated.mp4"
            save_video(video, output_file, fps=24, quality=10)
            print(f'[GPU {gpu_id}] Successfully saved video: {output_file}')

        except Exception as e:
            print(f'[GPU {gpu_id}] Error processing video {video_name}: {e}')

def main():
    # --- Global settings ---
    data_path = './data/'
    model_name = 'Wan2.2-Fun-5B-Control-Camera'

    data_name = data_path.split('/')[1]
    save_path = f'./output/{data_name}/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # --- GPU detection and data splitting ---
    if not torch.cuda.is_available():
        print("Error: No available CUDA GPU.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} available GPUs.")

    data = pd.read_csv(f'{data_path}/metadata.csv')
    print(f"Total {len(data)} videos to process.")

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
            
        p = mp.Process(target=process_videos_on_gpu, args=(gpu_id, data_chunks[gpu_id], data_path, save_path))
        p.start()
        processes.append(p)

    # Wait for all sub-processes to complete
    for p in processes:
        p.join()

    print("All videos have been processed.")


if __name__ == '__main__':
    main()
import os
import pandas as pd
import torch
import numpy as np
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import torch.multiprocessing as mp

def autoregressive_video_generation(
    gpu_id, 
    data_chunk, 
    data_path, 
    save_path, 
    num_cycles=3,
):
    """
    Autoregressive video generation process:
    1. Each clip uses the ground truth frame
    2. Generate 121 frames each time
    """
    # 1. Set the GPU to be used for this process
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Autoregressive process started, will run on this device: {device}")

    # 2. Load the Wan2.2 model
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ]
    )
    pipe.enable_vram_management()
    print(f"[GPU {gpu_id}] Wan2.2 model loaded.")

    # 3. Process each data sample
    for _, row in data_chunk.iterrows():
        video_name = row['video'].split('/')[-1]
        prompt = row['prompt']
        
        print(f'[GPU {gpu_id}] Start autoregressive generation: {video_name} | Prompt: {prompt}')
        print(f'[GPU {gpu_id}] Will generate {num_cycles} cycles, total {num_cycles * 120} frames') # not save the last frame
        
        try:
            # Read initial image
            input_image = VideoData(f"{data_path}/videos/{video_name}", height=704, width=1280)[0]
            
            current_image = input_image
            
            all_generated_frames = []
            
            # Autoregressive generation cycle
            for cycle in range(num_cycles):
                print(f"[GPU {gpu_id}] Start {cycle + 1}/{num_cycles} cycle")
                
                # Generate 121 frames video
                video_frames = pipe(
                    prompt=prompt,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    input_image=current_image,
                    height=704, width=1280,
                    num_frames=121,
                    seed=1, 
                    tiled=True,
                )
                
                # Save current cycle frames
                all_generated_frames.extend(video_frames[:-1])
                
                # If not the last cycle, use the last frame as the input image for the next cycle
                if cycle < num_cycles - 1:
                    current_image = video_frames[-1]  # Take the 120th frame (index=120)
            
            
            print(f"[GPU {gpu_id}] All cycles completed, final video total frames: {len(all_generated_frames)}")
            
            # Save final video
            output_file = f"{save_path}/{os.path.splitext(video_name)[0]}_autoregressive_{num_cycles}cycles.mp4"
            save_video(all_generated_frames, output_file, fps=24, quality=10)
            print(f'[GPU {gpu_id}] Successfully saved autoregressive video: {output_file}')

        except Exception as e:
            print(f'[GPU {gpu_id}] Error processing video {video_name}: {e}')
            import traceback
            traceback.print_exc()

def main():
    # --- Global settings ---
    data_path = './data'
    model_name = 'Wan2.2-TI2V-5B'
    data_name = data_path.split('/')[1]
    
    # Autoregressive parameters
    num_cycles = 3  # Number of cycles to generate
    
    save_path = f'./output/{data_name}/{model_name}-autoregressive-{num_cycles}cycles'
    os.makedirs(save_path, exist_ok=True)

    # --- GPU detection and data splitting ---
    if not torch.cuda.is_available():
        print("Error: No available CUDA GPU.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} available GPUs.")

    # Read the complete metadata.csv
    all_data = pd.read_csv(f'{data_path}/metadata.csv')
    
    # Only select the data of the clips as the main processing data
    data = all_data[all_data['video'].str.contains('_0.mp4')].copy()

    print(f"Selected {len(data)} clips to process (total {len(all_data)} clips).")
    print(f"Using autoregressive mode, each clip will generate {num_cycles} cycles, total {num_cycles * 120} frames")
    
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
        
        p = mp.Process(
            target=autoregressive_video_generation, 
            args=(
                gpu_id, 
                data_chunks[gpu_id],  
                data_path, 
                save_path, 
                num_cycles
            )
        )
    
        p.start()
        processes.append(p)

    # Wait for all sub-processes to complete
    for p in processes:
        p.join()

    print("All videos have been processed.")


if __name__ == '__main__':
    main()
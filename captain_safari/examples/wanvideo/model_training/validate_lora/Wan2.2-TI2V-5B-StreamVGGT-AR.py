import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import torch.multiprocessing as mp

# Add StreamVGGT imports
from diffsynth.streamvggt.streamvggt import StreamVGGTInference

def autoregressive_video_generation(
    gpu_id, 
    data_chunk, 
    model_path, 
    data_path, 
    save_path, 
    num_cycles=3,
    fps_interval=0.25,
    streamvggt_checkpoint_path=None
):
    """
    Autoregressive video generation process:
    1. The first clip uses ground truth memory
    2. The subsequent clips use StreamVGGT online generated memory
    3. Generate 121 frames each time, StreamVGGT frame sampling with fps_interval=0.25    
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
        ],
    )
    pipe.load_lora(pipe.dit, model_path, alpha=1)
    pipe.load_memory(pipe.dit, model_path)
    pipe.enable_vram_management()
    print(f"[GPU {gpu_id}] Wan2.2 model loaded.")

    # 3. Initialize StreamVGGT
    streamvggt_inferencer = StreamVGGTInference(checkpoint_path=streamvggt_checkpoint_path)
    print(f"[GPU {gpu_id}] StreamVGGT model loaded.")

    # 4. Process each data sample
    for _, row in data_chunk.iterrows():
        video_name = row['video'].split('/')[-1]
        prompt = row['prompt']
        initial_memory_path = row['memory']
        
        print(f'[GPU {gpu_id}] Start autoregressive generation: {video_name} | Prompt: {prompt}')
        print(f'[GPU {gpu_id}] Will generate {num_cycles} cycles, total {num_cycles * 120} frames')
        
        try:
            # Read initial memory (ground truth)
            initial_memory = np.load(os.path.join(data_path, initial_memory_path), allow_pickle=True) # (4, 782, 1024)
            current_memory = torch.tensor(initial_memory, dtype=torch.bfloat16).flatten(0, 2).unsqueeze(0) # torch.Size([1, 3128, 1024])
            
            # Read initial image
            input_image = VideoData(f"{data_path}/videos/{video_name}", height=704, width=1280)[0]
            
            current_image = input_image
            
            # Reset StreamVGGT state
            streamvggt_inferencer.reset_inference_state()
            
            # Create temporary directory
            os.makedirs("./tmp", exist_ok=True)
            
            all_generated_frames = []
            
            # Autoregressive generation cycle
            for cycle in range(num_cycles):
                print(f"[GPU {gpu_id}] Start {cycle + 1}/{num_cycles} cycle")
                
                # Generate 121 frames video
                video_frames = pipe(
                    prompt=prompt,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    memory=current_memory,
                    input_image=current_image,
                    height=704, width=1280,
                    num_frames=121,
                    seed=1, 
                    tiled=True,
                )
                
                # Save current cycle frames
                all_generated_frames.extend(video_frames[:-1])
                
                # If not the last cycle, use StreamVGGT to process the generated video to get new memory
                if cycle < num_cycles - 1:
                    # Create temporary video file
                    temp_video_path = f"./tmp/temp_video_cycle_{cycle}_{gpu_id}.mp4"
                    
                    # Resize video frames from 704*1280 to 720*1280
                    resized_frames = []
                    for frame in video_frames:
                        resized_frame = frame.resize((1280, 720), Image.LANCZOS)
                        resized_frames.append(resized_frame)
                    save_video(resized_frames, temp_video_path, fps=24, quality=10)
                    
                    
                    try:
                        # Use StreamVGGT to process the generated video
                        print(f"[GPU {gpu_id}] Use StreamVGGT to process the {cycle + 1} cycle video...")
                        global_memory, _ = streamvggt_inferencer.run_continuous_inference(
                            temp_video_path, 
                            fps_interval=fps_interval
                        )
                        
                        # Convert memory format to be used for the next cycle
                        # global_memory shape: [N, H, W, 1024] -> [1, H*W, 1024], only take the last memory as condition
                        global_memory_tensor = torch.tensor(global_memory[-1], dtype=torch.bfloat16) # [1, H*W, 1024]
                        current_memory = global_memory_tensor.flatten(0, -2).unsqueeze(0)  # [1, H*W, 1024]
                        
                        # Use the last frame as the input image for the next cycle
                        current_image = video_frames[-1]  # Take the 120th frame (index=120)
                        
                        print(f"[GPU {gpu_id}] StreamVGGT processing completed, new memory shape: {current_memory.shape}") # [1, H*W, 1024]
                        
                    finally:
                        # Clean up temporary files
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
            
            
            print(f"[GPU {gpu_id}] All cycles completed, final video total frames: {len(all_generated_frames)}")
            
            # Save final video
            output_file = f"{save_path}/{os.path.splitext(video_name)[0]}_autoregressive-StreamVGGT-{num_cycles}cycles.mp4"
            save_video(all_generated_frames, output_file, fps=24, quality=10)
            print(f'[GPU {gpu_id}] Successfully saved autoregressive video: {output_file}')

        except Exception as e:
            print(f'[GPU {gpu_id}] Error processing video {video_name}: {e}')
            import traceback
            traceback.print_exc()

def main():
    # --- Global settings ---
    data_path = './data'
    model_path = './models/train/Wan2.2-TI2V-5B-StreamVGGT_lora/epoch-4.safetensors'
    
    # Autoregressive parameters
    num_cycles = 3  # Number of cycles to generate
    fps_interval = 0.25  # StreamVGGT frame sampling interval
    streamvggt_checkpoint_path = None  # StreamVGGT checkpoint path, None means download from HuggingFace
    
    epoch = model_path.split('/')[4].split('.')[0]
    model_name = model_path.split('/')[3]
    data_name = data_path.split('/')[1]
    
    save_path = f'./output/{data_name}/{model_name}-{epoch}-autoregressive-StreamVGGT-{num_cycles}cycles'
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
    
    print(f"Using autoregressive mode, each video will generate {num_cycles} cycles, total {num_cycles * 120} frames")
    print(f"StreamVGGT frame sampling interval: {fps_interval} seconds")

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
                model_path, 
                data_path, 
                save_path, 
                num_cycles, 
                fps_interval, 
                streamvggt_checkpoint_path
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
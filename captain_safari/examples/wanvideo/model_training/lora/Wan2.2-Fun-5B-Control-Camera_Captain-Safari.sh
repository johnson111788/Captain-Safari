export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=12345

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path ./data/ \
  --dataset_metadata_path ./data/metadata.csv \
  --height 704 \
  --width 1280 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "PAI/Wan2.2-Fun-5B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.2-Fun-5B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-Fun-5B-Control-Camera:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Fun-5B-Control-Camera_Captain-Safari" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --save_checkpoint_steps 600 \
  --extra_inputs "input_image,memory,intrinsic_query,extrinsic_query,intrinsic_key,extrinsic_key,extrinsic_clip,intrinsic_clip" \
  --data_file_keys "image,video,memory,intrinsic_query,extrinsic_query,intrinsic_key,extrinsic_key,extrinsic_clip,intrinsic_clip" \
  --use_memory_retrieval \
  --use_memory_cross_attn \
  --load_memory_retriever_path "checkpoints/memory-retriever-warmup/final_step_5000.pt" \
  --max_grad_norm 1.0 \
  --wandb_project "Captain-Safari" \
  --wandb_run_name "Wan2.2-Fun-5B-Control-Camera_Captain-Safari"
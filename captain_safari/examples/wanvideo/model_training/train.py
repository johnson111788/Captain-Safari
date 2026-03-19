import torch, os, json, random
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        use_memory_cross_attn=False,
        use_memory_retrieval=False,
        load_memory_retriever_path=None,
        resume_from_checkpoint=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Set memory cross-attention mode
        self.pipe.dit.use_memory_cross_attn = use_memory_cross_attn
        self.pipe.dit.use_memory_retrieval = use_memory_retrieval
        print(f"set use_memory_cross_attn: {use_memory_cross_attn}")
        print(f"set use_memory_retrieval: {use_memory_retrieval}")
        
        # Load pre-trained memory retriever weights if provided
        if load_memory_retriever_path is not None and use_memory_retrieval:
            print(f"load memory retriever weights: {load_memory_retriever_path}")
            success = self.pipe.load_memory_retriever(self.pipe.dit, load_memory_retriever_path)
            if success:
                print("✅ Memory retriever weights loaded successfully")
            else:
                print("❌ Memory retriever weights loaded failed")
        elif load_memory_retriever_path is not None and not use_memory_retrieval:
            print("⚠️  provided memory retriever weights path, but use_memory_retrieval is not enabled")
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = getattr(self.pipe, lora_base_model)
            
            # if memory cross-attention or memory retrieval is enabled, exclude related layers
            target_modules = lora_target_modules.split(",")
            should_filter = ((hasattr(model, 'use_memory_cross_attn') and model.use_memory_cross_attn) or 
                           (hasattr(model, 'use_memory_retrieval') and model.use_memory_retrieval))
            
            if should_filter:
                # collect all target layers that do not contain memory_cross_attn and memory_retriever
                filtered_modules = []
                excluded_patterns = ['memory_cross_attn', 'memory_retriever']
                
                for name, module in model.named_modules():
                    if hasattr(module, 'weight'):
                        # check if any excluded pattern is contained in the name
                        should_exclude = any(pattern in name for pattern in excluded_patterns)
                        if not should_exclude:
                            for target in target_modules:
                                if name.endswith(f'.{target}'):
                                    filtered_modules.append(name)
                target_modules = filtered_modules if filtered_modules else target_modules
                print(f"exclude memory related modules (memory_cross_attn, memory_retriever), LoRA target: {len(target_modules)}/300 modules")
            
            model = self.add_lora_to_model(
                model,
                target_modules=target_modules,
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)

        # Load checkpoint if provided (for resuming training)
        # important: must load checkpoint after add_lora_to_model!
        if resume_from_checkpoint is not None:
            print(f"resume training from checkpoint: {resume_from_checkpoint}")
            
            # the checkpoint saved for training is a complete state_dict, need to load_state_dict directly
            # instead of using load_lora (that is used for inference)
            from diffsynth import load_state_dict
            checkpoint_state_dict = load_state_dict(resume_from_checkpoint, torch_dtype=torch.bfloat16, device="cpu")
            
            # note: the prefix "pipe.dit." has been removed when saving, so directly use checkpoint_state_dict
            # use strict=False, only load existing parameters
            missing_keys, unexpected_keys = self.pipe.dit.load_state_dict(checkpoint_state_dict, strict=False)
            
            if missing_keys:
                print(f"   Missing keys: {len(missing_keys)} (this is normal, parameters that are not trained are not in the checkpoint)")
            if unexpected_keys:
                print(f"   ⚠️ Unexpected keys: {unexpected_keys}")

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        # Unfreeze memory_emb (only if memory training is requested)
        if "memory" in self.extra_inputs:
            for p in self.pipe.dit.memory_emb.parameters():
                p.requires_grad_(True)
            print("enable memory_emb training")
            
            # Unfreeze memory cross-attention if enabled
            if hasattr(self.pipe.dit, 'use_memory_cross_attn') and self.pipe.dit.use_memory_cross_attn:
                for block in self.pipe.dit.blocks:
                    if hasattr(block, 'memory_cross_attn'):
                        for p in block.memory_cross_attn.parameters():
                            p.requires_grad_(True)
                        for p in block.norm_memory.parameters():
                            p.requires_grad_(True)
                print("enable memory_cross_attn training")
            
            # Unfreeze memory retrieval if enabled
            if hasattr(self.pipe.dit, 'use_memory_retrieval') and self.pipe.dit.use_memory_retrieval:
                if hasattr(self.pipe.dit, 'memory_retriever'):
                    for p in self.pipe.dit.memory_retriever.parameters():
                        p.requires_grad_(True)
                    print("enable memory_retrieval training")
        else:
            print("skip memory_emb training (not in extra_inputs)")


    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        # inputs_posi = {"prompt": data["prompt"] if random.random() < 0.8 else data["short_prompt"]}
        
        # process pre-encoded data
        # if using pre-encoded data, data will have "prompt_embedding" and "video_latent" fields
        if "prompt_embedding" in data:
            inputs_posi = {"prompt": data["prompt_embedding"]}
        else:
            inputs_posi = {"prompt": data["prompt"]} # TODO: Don't forget to change this back to the original
        inputs_nega = {}
        
        # check if using pre-encoded video latents
        if "video_latent" in data:
            # use pre-encoded video latents
            input_video = data["video_latent"]
            # infer original video size from latent shape
            # latent shape: [B, C, T, H, W]
            # original video size = latent size * upsampling_factor (16)
            height = input_video.shape[3] * 16
            width = input_video.shape[4] * 16
            # T latents = 1 + (num_frames - 1) // 4
            num_frames = 1 + (input_video.shape[2] - 1) * 4
        else:
            # use original video
            input_video = data["video"]
            height = data["video"][0].size[1]
            width = data["video"][0].size[0]
            num_frames = len(data["video"])
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": input_video,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                # process pre-encoded input_image
                if "image_y_latent" in data and "image_single_latent" in data:
                    # use pre-encoded image latents
                    inputs_shared["input_image"] = {
                        'y': data["image_y_latent"],
                        'single': data["image_single_latent"]
                    }
                elif "video" in data and isinstance(data["video"], list):
                    # use original video's first frame
                    inputs_shared["input_image"] = data["video"][0]
                else:
                    # skip (no original video in pre-encoded mode)
                    print("Warning: input_image cannot be obtained, skip")
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "memory":
                if "memory" in data:
                    inputs_shared["memory"] = data["memory"]
                else:
                    print("Warning: extra_inputs contains memory but data does not have memory field")
            elif extra_input in ["intrinsic_query", "extrinsic_query", "intrinsic_key", "extrinsic_key", "extrinsic_clip", "intrinsic_clip"]:
                if extra_input in data:
                    inputs_shared[extra_input] = data[extra_input]
                else:
                    print(f"Warning: extra_inputs contains {extra_input} but data does not have {extra_input} field")
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data) # data['memory'].shape: (4, 782, 1024)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = VideoDataset(args=args)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        use_memory_cross_attn=args.use_memory_cross_attn,
        use_memory_retrieval=args.use_memory_retrieval,
        use_nearest_memory_retrieval=args.use_nearest_memory_retrieval,
        load_memory_retriever_path=args.load_memory_retriever_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_checkpoint_steps=args.save_checkpoint_steps,
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
    )

import os
import cv2
import torch
import numpy as np
import sys
import tempfile
import shutil
from typing import Union, List, Dict, Optional
import glob
import argparse
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from streamvggt.visual_util import predictions_to_glb
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import unproject_depth_map_to_point_map


class StreamVGGTInference:
    def __init__(self, checkpoint_path: Optional[str] = None, feature_only: bool = False, aux_feature: bool = False, device_id: Optional[int] = None):
        """
        Initialize StreamVGGT model for inference.
        
        Args:
            checkpoint_path: Path to local checkpoint. If None, downloads from HuggingFace.
            feature_only: Whether to run in feature-only mode
            aux_feature: Whether to use auxiliary features
            device_id: Specific GPU device ID to use. If None, uses current device.
        """
        if device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            self.device = f"cuda:{device_id}"
            device_name = torch.cuda.get_device_name(device_id)
            print(f"Using device: {self.device} ({device_name})")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
        
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU which may be slow")
        
        self.feature_only = feature_only
        self.aux_feature = aux_feature
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path: Optional[str] = None) -> StreamVGGT:
        """Load and initialize the StreamVGGT model."""
        print("Initializing and loading StreamVGGT model...")
        
        model = StreamVGGT(feature_only=self.feature_only, aux_feature=self.aux_feature)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading local checkpoint from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        else:
            print("Loading checkpoint from Hugging Face...")
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="lch01/StreamVGGT",
                filename="checkpoints.pth",
                revision="main",
                force_download=True
            )
            ckpt = torch.load(path, map_location="cpu")
        
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model = model.to(self.device)
        del ckpt
        
        return model
        
    def _extract_frames_from_video(self, video_path: str, fps_interval: float = 1.0) -> List[np.ndarray]:
        """
        Extract frames from video at specified interval.
        
        Args:
            video_path: Path to video file
            fps_interval: Extract one frame every fps_interval seconds
            
        Returns:
            List of frames as numpy arrays (BGR format)
        """
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * fps_interval)
        
        frames = []
        count = 0
        
        while True:
            ret, frame = vs.read()
            if not ret:
                break
                
            count += 1
            if count % frame_interval == 0:
                frames.append(frame)
                
        vs.release()
        return frames
        
    def _save_frames_to_temp_dir(self, frames: List[np.ndarray]) -> str:
        """Save frames to temporary directory and return the path."""
        temp_dir = tempfile.mkdtemp()
        images_dir = os.path.join(temp_dir, "images")
        os.makedirs(images_dir)
        
        for i, frame in enumerate(frames):
            frame_path = os.path.join(images_dir, f"{i:06d}.png")
            cv2.imwrite(frame_path, frame)
            
        return temp_dir
        
    def _process_input(self, input_data: Union[str, List[str], List[np.ndarray]], fps_interval: float = 1.0) -> str:
        """
        Process input data and return path to temporary directory with images.
        
        Args:
            input_data: Can be:
                - String path to video file
                - List of string paths to image files  
                - List of numpy arrays (image frames)
            fps_interval: Frame sampling interval for videos
                
        Returns:
            Path to temporary directory containing processed images
        """
        if isinstance(input_data, str):
            # Single video file
            if input_data.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                frames = self._extract_frames_from_video(input_data, fps_interval)
                return self._save_frames_to_temp_dir(frames)
            else:
                raise ValueError(f"Unsupported file format: {input_data}")
                
        elif isinstance(input_data, list):
            if len(input_data) == 0:
                raise ValueError("Empty input list")
                
            if isinstance(input_data[0], str):
                # List of image file paths
                temp_dir = tempfile.mkdtemp()
                images_dir = os.path.join(temp_dir, "images")
                os.makedirs(images_dir)
                
                for i, img_path in enumerate(input_data):
                    if not os.path.exists(img_path):
                        raise ValueError(f"Image file not found: {img_path}")
                    dst_path = os.path.join(images_dir, f"{i:06d}{os.path.splitext(img_path)[1]}")
                    shutil.copy2(img_path, dst_path)
                    
                return temp_dir
                
            elif isinstance(input_data[0], np.ndarray):
                # List of numpy arrays (frames)
                return self._save_frames_to_temp_dir(input_data)
            else:
                raise ValueError("Unsupported input format in list")
        else:
            raise ValueError("Input must be video path, list of image paths, or list of numpy arrays")

    def run_inference(self, input_data: Union[str, List[str], List[np.ndarray]], 
                     fps_interval: float = 1.0, export_memory: bool = False) -> Dict:
        """
        Run StreamVGGT inference on input data.
        
        Args:
            input_data: Input video file path, list of image paths, or list of image arrays
            fps_interval: Frame sampling interval for videos
            export_memory: Whether to export memory features along with predictions
            
        Returns:
            Dictionary containing predictions and optionally memory features
        """
        # Process input to get temporary directory with images
        temp_dir = self._process_input(input_data, fps_interval)
        
        try:
            # Run model inference
            results = self._run_model_inference(temp_dir, export_memory=export_memory)
            return results
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    def _run_model_inference(self, target_dir: str, export_memory: bool = False) -> Dict:
        """Core model inference logic."""
        print(f"Processing images from {target_dir}")
        
        # Load and preprocess images
        image_names = glob.glob(os.path.join(target_dir, "images", "*"))
        image_names = sorted(image_names)
        print(f"Found {len(image_names)} images")
        
        if len(image_names) == 0:
            raise ValueError("No images found in input")
            
        images = load_and_preprocess_images(image_names).to(self.device)
        print(f"Preprocessed images shape: {images.shape}")
        
        # Prepare frames for model
        frames = []
        for i in range(images.shape[0]):
            image = images[i].unsqueeze(0)
            frame = {"img": image}
            frames.append(frame)
            
        # Run inference
        print(f"Running inference with export_memory={export_memory}...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                if export_memory:
                    # Return tuple: (output, aggregated_tokens, pts3d_features)
                    output, aggregated_tokens, pts3d_features = self.model.inference(frames, export_memory=True)
                else:
                    # Return only output
                    output = self.model.inference(frames, export_memory=False)
       
        # Extract results from output.ress
        all_pts3d = []
        all_conf = []
        all_depth = []
        all_depth_conf = []
        all_camera_pose = []
        
        for res in output.ress:
            all_pts3d.append(res['pts3d_in_other_view'].squeeze(0))
            all_conf.append(res['conf'].squeeze(0))
            all_depth.append(res['depth'].squeeze(0))
            all_depth_conf.append(res['depth_conf'].squeeze(0))
            all_camera_pose.append(res['camera_pose'].squeeze(0))
            
        # Build predictions dictionary
        predictions = {}
        predictions["images"] = images.cpu().numpy()
        predictions["world_points"] = torch.stack(all_pts3d, dim=0).cpu().numpy()
        predictions["world_points_conf"] = torch.stack(all_conf, dim=0).cpu().numpy()
        predictions["depth"] = torch.stack(all_depth, dim=0).cpu().numpy()
        predictions["depth_conf"] = torch.stack(all_depth_conf, dim=0).cpu().numpy()
        predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0).cpu().numpy()
        
        print("World points shape:", predictions["world_points"].shape)
        print("Depth map shape:", predictions["depth"].shape)
        print("Pose encoding shape:", predictions["pose_enc"].shape)
        
        # Convert pose encoding to extrinsic and intrinsic matrices
        print("Converting pose encoding to camera matrices...")
        pose_enc_tensor = torch.from_numpy(predictions["pose_enc"]).to(self.device)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc_tensor.unsqueeze(0) if pose_enc_tensor.ndim == 2 else pose_enc_tensor, 
            images.shape[-2:]
        )
        
        predictions["extrinsic"] = extrinsic.squeeze(0).cpu().numpy()
        predictions["intrinsic"] = intrinsic.squeeze(0).cpu().numpy() if intrinsic is not None else None
        
        # For now, use the same world points (can implement depth unprojection if needed)
        predictions["world_points_from_depth"] = predictions["world_points"]
        
        # If export_memory is True, add memory features to results
        if export_memory:
            predictions["aggregated_tokens"] = aggregated_tokens
            if pts3d_features is not None:
                predictions["pts3d_features"] = pts3d_features
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        return predictions
        
    def create_3d_visualization(self, predictions: Dict, output_path: str, 
                               conf_thres: float = 50.0,
                               frame_filter: str = "All",
                               mask_black_bg: bool = False,
                               mask_white_bg: bool = False, 
                               show_cam: bool = True,
                               mask_sky: bool = False,
                               prediction_mode: str = "Depthmap and Camera Branch") -> str:
        """
        Create 3D GLB visualization from predictions.
        
        Args:
            predictions: Predictions dictionary from run_inference
            output_path: Path to save GLB file
            conf_thres: Confidence threshold for filtering points
            frame_filter: Which frames to show ("All" or specific frame)
            mask_black_bg: Filter black background points
            mask_white_bg: Filter white background points
            show_cam: Show camera positions
            mask_sky: Filter sky points
            prediction_mode: Prediction mode for visualization
            
        Returns:
            Path to saved GLB file
        """
        # Create temporary directory for GLB generation
        temp_dir = tempfile.mkdtemp()
        
        try:
            glbscene = predictions_to_glb(
                predictions,
                conf_thres=conf_thres,
                filter_by_frames=frame_filter,
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                show_cam=show_cam,
                mask_sky=mask_sky,
                target_dir=temp_dir,
                prediction_mode=prediction_mode,
            )
            glbscene.export(file_obj=output_path)
            return output_path
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def save_predictions(self, predictions, save_path):
        """Save predictions to file."""
        if isinstance(predictions, dict):
            # Save as NPZ for dictionary predictions
            np.savez_compressed(save_path, **predictions)
        else:
            # Save as NPY for single array
            np.save(save_path, predictions)
        print(f"Predictions saved to {save_path}")

    def load_memory(self, path):
        """Load memory from saved file."""
        memory = np.load(path, allow_pickle=True).item()
        
        for frame_id in memory:
            frame_memory = memory[frame_id]
            if 'aggregated_tokens_global' in frame_memory:
                global_memory = torch.tensor(frame_memory['aggregated_tokens_global'])
            else:
                global_memory = torch.tensor(frame_memory['aggregated_tokens'])[:, :, :, -1024:]
        
        return global_memory


def gradio_multi_method_visualization(base_folders=None):
    """
    Launch Gradio interface for viewing GLB models from multiple methods side by side.
    
    Args:
        base_folders (list, optional): List of [ours_folder, baseline_folder, gt_folder]
    """
    import gradio as gr
    from pathlib import Path
    
    # Use default folders if none provided
    if base_folders is None:
        base_folders = [
            "reconstructions/ours",
            "reconstructions/baseline", 
            "reconstructions/gt"
        ]
    
    method_names = ["ours", "baseline", "gt"]
    
    def find_common_video_ids():
        """Find video IDs that exist in all three folders."""
        video_ids_per_folder = []
        
        for folder in base_folders:
            if not os.path.exists(folder):
                print(f"Warning: Folder {folder} does not exist")
                return []
                
            glb_files = list(Path(folder).glob("*.glb"))
            
            # Extract video IDs based on folder type
            video_ids = set()
            folder_name = os.path.basename(folder.rstrip('/'))
            
            for glb_file in glb_files:
                filename = glb_file.stem  # Remove .glb extension
                
                if folder_name in ["ours", "baseline"]:
                    # ours and baseline files: video_id_1_autoregressive_3cycles.glb
                    if "_1_autoregressive_3cycles" in filename:
                        video_id = filename.replace("_1_autoregressive_3cycles", "")
                        video_ids.add(video_id)
                else:
                    # gt files: video_id.glb
                    video_ids.add(filename)
            
            video_ids_per_folder.append(video_ids)
        
        # Find intersection of all sets
        if video_ids_per_folder:
            common_video_ids = set.intersection(*video_ids_per_folder)
            return sorted(list(common_video_ids))
        else:
            return []
    
    def load_glb_models(video_id):
        """Load GLB models for a specific video ID from all methods."""
        if not video_id:
            return None, None, None, "Please select a video ID"
        
        glb_paths = []
        status_messages = []
        
        for folder, method_name in zip(base_folders, method_names):
            # Construct filename based on method
            if method_name in ["ours", "baseline"]:
                filename = f"{video_id}_1_autoregressive_3cycles.glb"
            else:
                filename = f"{video_id}.glb"
                
            glb_path = os.path.join(folder, filename)
            
            if os.path.exists(glb_path):
                glb_paths.append(glb_path)
                file_size = os.path.getsize(glb_path) / (1024 * 1024)  # MB
                status_messages.append(f"✓ {method_name}: {file_size:.2f} MB")
            else:
                glb_paths.append(None)
                status_messages.append(f"✗ {method_name}: Not found")
        
        status = "\n".join(status_messages)
        return glb_paths[0], glb_paths[1], glb_paths[2], status
    
    def update_info(video_id):
        """Update information about the selected video."""
        if not video_id:
            return "No video selected"
        
        info_lines = [f"Selected Video: {video_id}"]
        
        # Check file existence and add details
        for folder, method_name in zip(base_folders, method_names):
            if method_name in ["ours", "baseline"]:
                filename = f"{video_id}_1_autoregressive_3cycles.glb"
            else:
                filename = f"{video_id}.glb"
                
            glb_path = os.path.join(folder, filename)
            
            if os.path.exists(glb_path):
                # Get file modification time
                import time
                mtime = os.path.getmtime(glb_path)
                mod_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
                info_lines.append(f"{method_name}: Modified {mod_time}")
            else:
                info_lines.append(f"{method_name}: Missing")
        
        return "\n".join(info_lines)
    
    # Find common video IDs
    print("Finding common video IDs across all methods...")
    common_video_ids = find_common_video_ids()
    
    if not common_video_ids:
        print("No common video IDs found across all folders.")
        print("Please check your folder structure:")
        for i, folder in enumerate(base_folders):
            print(f"  {method_names[i]}: {folder}")
        return
    
    print(f"Found {len(common_video_ids)} common video IDs")
    
    # Create Gradio interface
    with gr.Blocks(
        title="Multi-Method 3D Reconstruction Comparison", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {max-width: 1400px !important}
        .model3d {height: 450px !important}
        """
    ) as demo:
        
        gr.Markdown("# 🎥 Multi-Method 3D Reconstruction Comparison")
        gr.Markdown("Compare point cloud reconstructions from **Ours**, **Baseline**, and **Ground Truth** methods")
        
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                # Video selection panel
                gr.Markdown("### 📂 Video Selection")
                
                video_dropdown = gr.Dropdown(
                    choices=common_video_ids,
                    label="Video ID",
                    value=common_video_ids[0] if common_video_ids else None,
                    interactive=True,
                    elem_id="video_selector"
                )
                
                load_btn = gr.Button(
                    "🔄 Load 3D Models", 
                    variant="primary", 
                    size="lg",
                    elem_id="load_button"
                )
                
                # Information panel
                with gr.Accordion("📊 Model Information", open=True):
                    info_text = gr.Textbox(
                        label="File Details", 
                        lines=4,
                        interactive=False,
                        elem_id="info_display"
                    )
                    
                    status_text = gr.Textbox(
                        label="Loading Status",
                        lines=4,
                        interactive=False,
                        elem_id="status_display"
                    )
                
                # Statistics (placeholder for future enhancement)
                with gr.Accordion("📈 Quick Stats", open=False):
                    gr.Markdown(f"""
                    **Available Videos:** {len(common_video_ids)}  
                    **Methods:** Ours, Baseline, Ground Truth  
                    **Folders:**
                    - Ours: `{base_folders[0]}`
                    - Baseline: `{base_folders[1]}`  
                    - GT: `{base_folders[2]}`
                    """)
        
        # 3D Viewers section
        gr.Markdown("## 🎮 3D Point Cloud Comparison")
        
        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("### 🔴 **Ours**")
                ours_viewer = gr.Model3D(
                    label="Our Method",
                    height=450,
                    camera_position=[0, 0, 2],
                    elem_id="ours_viewer"
                )
            
            with gr.Column():
                gr.Markdown("### 🔵 **Baseline**")
                baseline_viewer = gr.Model3D(
                    label="Baseline Method",
                    height=450,
                    camera_position=[0, 0, 2],
                    elem_id="baseline_viewer"
                )
            
            with gr.Column():
                gr.Markdown("### 🟢 **Ground Truth**")
                gt_viewer = gr.Model3D(
                    label="Ground Truth",
                    height=450,
                    camera_position=[0, 0, 2],
                    elem_id="gt_viewer"
                )
        
        # Usage instructions
        with gr.Accordion("🎯 How to Use", open=False):
            gr.Markdown("""
            ### Navigation Controls:
            - **🖱️ Mouse Drag**: Rotate the 3D view
            - **🔄 Mouse Wheel**: Zoom in/out  
            - **⇧ Shift + Drag**: Pan the view
            - **🖱️ Double Click**: Reset to default view
            
            ### Comparison Tips:
            - Select different video IDs to compare reconstructions
            - Use the same viewing angle across all three models for better comparison
            - Check the loading status to ensure all models loaded successfully
            """)
        
        # Event handlers
        load_btn.click(
            fn=load_glb_models,
            inputs=[video_dropdown],
            outputs=[ours_viewer, baseline_viewer, gt_viewer, status_text]
        )
        
        video_dropdown.change(
            fn=update_info,
            inputs=[video_dropdown],
            outputs=[info_text]
        )
        
        # Auto-load on startup
        demo.load(
            fn=update_info,
            inputs=[video_dropdown],
            outputs=[info_text]
        )
        
        demo.load(
            fn=load_glb_models,
            inputs=[video_dropdown],
            outputs=[ours_viewer, baseline_viewer, gt_viewer, status_text]
        )
    
    return demo


def distribute_videos_across_gpus(videos, num_gpus, args):
    """
    Distribute videos across GPUs with load balancing.
    
    Args:
        videos: List of video files to process
        num_gpus: Number of available GPUs
        args: Command line arguments
        
    Returns:
        List of (gpu_id, video_batch) tuples
    """
    # Round-robin distribution for better load balancing
    video_batches = [[] for _ in range(num_gpus)]
    
    # Use round-robin distribution
    for i, video in enumerate(videos):
        gpu_id = i % num_gpus
        video_batches[gpu_id].append(video)
    
    # Convert to (gpu_id, batch) tuples, filtering out empty batches
    result = []
    for gpu_id, batch in enumerate(video_batches):
        if batch:
            result.append((gpu_id, batch))
    
    return result


def process_videos_on_gpu(gpu_id, video_batch, args, checkpoint_path):
    """Process a batch of videos on a specific GPU."""
    try:
        # Import torch here to ensure proper initialization in spawned process
        import torch
        
        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available in GPU {gpu_id} process")
        
        # Directly set the device to the specified GPU ID
        torch.cuda.set_device(gpu_id)
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        
        # Initialize model for this specific GPU
        print(f"GPU {gpu_id}: Initializing model on {device_name} (device: cuda:{device})...")
        inferencer = StreamVGGTInference(checkpoint_path, feature_only=False, aux_feature=True, device_id=gpu_id)
        
        processed_count = 0
        error_count = 0
        
        for i, video in enumerate(video_batch):
            video_id = video.split('.')[0]
            aggregated_tokens_path = f"{args.output}/{video_id}_aggregated_tokens.npy"
            pts3d_features_path = f"{args.output}/{video_id}_pts3d_features.npy"
            
            # Progress reporting
            print(f"GPU {gpu_id}: Processing {video} ({i+1}/{len(video_batch)})")
            
            # Skip if already processed
            if os.path.exists(aggregated_tokens_path) and os.path.exists(pts3d_features_path):
                print(f"GPU {gpu_id}: Skipping {video}, already processed.")
                continue
                
            video_path = os.path.join(args.video_root, video)
            
            try:
                # Run inference with export_memory=True
                predictions = inferencer.run_inference(video_path, fps_interval=args.fps_interval, export_memory=True)
                
                # Save memory features
                if 'aggregated_tokens' in predictions:
                    inferencer.save_predictions(predictions['aggregated_tokens'], aggregated_tokens_path)
                
                if 'pts3d_features' in predictions:
                    inferencer.save_predictions(predictions['pts3d_features'], pts3d_features_path)
                
                # Save additional outputs
                pointcloud_path = f"{args.output}/{video_id}_pointcloud.npy"
                extrinsic_path = f"{args.output}/{video_id}_extrinsic.npy"
                intrinsic_path = f"{args.output}/{video_id}_intrinsic.npy"
                
                np.save(pointcloud_path, predictions['world_points'])
                np.save(extrinsic_path, predictions['extrinsic'])
                np.save(intrinsic_path, predictions['intrinsic'])
                
                processed_count += 1
                del predictions
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"GPU {gpu_id}: Error processing {video}: {e}")
                error_count += 1
                continue
        
        return gpu_id, processed_count, error_count
        
    except Exception as e:
        print(f"GPU {gpu_id}: Fatal error in process: {e}")
        return gpu_id, 0, len(video_batch)


def run_export_memory(args, inferencer):
    """Run memory export mode with multi-GPU parallel processing."""
    print("Running memory export mode with multi-GPU support...")
    
    # Get list of videos to process
    videos = sorted(os.listdir(args.video_root))
    print(f"Total videos found: {len(videos)}")
    
    # Filter out already processed videos
    videos = [video for video in videos if 
              (not os.path.exists(f"{args.output}/{video.split('.')[0]}_aggregated_tokens.npy")) and 
              video.endswith('.mp4')]
    print(f"Videos to process: {len(videos)}")
    
    if len(videos) == 0:
        print("No videos need processing.")
        return
    
    # Handle job distribution for batch processing
    if args.job is not None and args.total is not None:
        start_idx = args.job * len(videos) // args.total
        end_idx = (args.job + 1) * len(videos) // args.total
        videos = videos[start_idx:end_idx]
        print(f"Processing videos {start_idx} to {end_idx} (job {args.job}/{args.total})")
    
    # Check for multi-GPU setup and user preferences
    num_gpus = torch.cuda.device_count()
    if args.force_single_gpu or not args.use_multi_gpu or num_gpus <= 1:
        if args.force_single_gpu:
            print("Forced single GPU mode, using sequential processing...")
        elif not args.use_multi_gpu:
            print("Multi-GPU disabled by user, using sequential processing...")
        else:
            print("Single GPU detected, using sequential processing...")
        return run_export_memory_sequential(args, inferencer, videos)
    
    print(f"Multi-GPU setup detected: {num_gpus} GPUs available")
    
    # Distribute videos across GPUs with better load balancing
    video_batches = distribute_videos_across_gpus(videos, num_gpus, args)
    
    for gpu_id, batch in video_batches:
        print(f"GPU {gpu_id}: {len(batch)} videos assigned")
    
    # Run parallel processing
    print(f"Starting parallel processing on {len(video_batches)} GPUs...")
    start_time = time.time()
    
    # Use multiprocessing to run on different GPUs
    # Note: spawn method is required for CUDA compatibility
    with mp.get_context('spawn').Pool(processes=len(video_batches)) as pool:
        # Create tasks for each GPU
        tasks = []
        for gpu_id, batch in video_batches:
            task = pool.apply_async(process_videos_on_gpu, 
                                  (gpu_id, batch, args, args.checkpoint_path))
            tasks.append(task)
        
        # Wait for all processes to complete and collect results
        total_processed = 0
        total_errors = 0
        
        for task in tasks:
            gpu_id, processed, errors = task.get()
            total_processed += processed
            total_errors += errors
            print(f"GPU {gpu_id} completed: {processed} processed, {errors} errors")
    
    end_time = time.time()
    print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
    print(f"Total processed: {total_processed}, Total errors: {total_errors}")


def run_export_memory_sequential(args, inferencer, videos):
    """Sequential processing fallback for single GPU."""
    print("Running sequential processing...")
    
    for video in tqdm(videos, desc="Processing videos"):
        video_id = video.split('.')[0]
        aggregated_tokens_path = f"{args.output}/{video_id}_aggregated_tokens.npy"
        pts3d_features_path = f"{args.output}/{video_id}_pts3d_features.npy"
        
        if os.path.exists(aggregated_tokens_path) and os.path.exists(pts3d_features_path):
            print(f"Skipping {video}, memory already processed.")
            continue
            
        video_path = os.path.join(args.video_root, video)
        try:
            # Run inference with export_memory=True
            predictions = inferencer.run_inference(video_path, fps_interval=args.fps_interval, export_memory=True)
            
            # Save memory features
            if 'aggregated_tokens' in predictions:
                inferencer.save_predictions(predictions['aggregated_tokens'], aggregated_tokens_path)
            
            if 'pts3d_features' in predictions:
                inferencer.save_predictions(predictions['pts3d_features'], pts3d_features_path)
            
            pointcloud_path = f"{args.output}/{video_id}_pointcloud.npy"
            extrinsic_path = f"{args.output}/{video_id}_extrinsic.npy"
            intrinsic_path = f"{args.output}/{video_id}_intrinsic.npy"
            
            np.save(pointcloud_path, predictions['world_points'])
            np.save(extrinsic_path, predictions['extrinsic'])
            np.save(intrinsic_path, predictions['intrinsic'])
            
            del predictions
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {video}: {e}")
            continue


def run_reconstruction(args, inferencer):
    """Run reconstruction mode."""
    print("Running reconstruction mode...")
    
    for video_id in tqdm(os.listdir(args.video_root)):
        video_path = os.path.join(args.video_root, video_id)
        video_name = video_id.split('.')[0]
        
        output_glb = f"{args.output}/{video_name}.glb"
        pointcloud_path = f"{args.output}/{video_name}_pointcloud.npy"
        extrinsic_path = f"{args.output}/{video_name}_extrinsic.npy"
        intrinsic_path = f"{args.output}/{video_name}_intrinsic.npy"
        
        if (os.path.exists(output_glb) and 
            os.path.exists(pointcloud_path) and 
            os.path.exists(extrinsic_path) and
            os.path.exists(intrinsic_path)):
            print(f"Skipping {video_id}, all outputs already exist.")
            continue
            
        try:
            # Run inference without memory export
            predictions = inferencer.run_inference(video_path, fps_interval=args.fps_interval, export_memory=False)
            
            # Create 3D visualization
            inferencer.create_3d_visualization(predictions, output_glb)
            print(f"Created reconstruction: {output_glb}")
            
            # Save point cloud and camera separately
            np.save(pointcloud_path, predictions['world_points'])
            np.save(extrinsic_path, predictions['extrinsic'])
            np.save(intrinsic_path, predictions['intrinsic'])
            print(f"Saved point cloud: {pointcloud_path}")
            print(f"Saved camera: {extrinsic_path}")
            print(f"Saved intrinsic: {intrinsic_path}")
            
            del predictions
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    # This must be done before any CUDA operations in multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # start method has already been set
        pass
        
    parser = argparse.ArgumentParser(description="StreamVGGT Inference Script")
    
    # Mode selection
    parser.add_argument('--mode', type=str, 
                       choices=['export_memory', 'reconstruction', 'visualization'],  # Yu-Cheng: here to visualize the results
                       default='export_memory', help='Mode to run')
    
    # Job distribution (for export_memory mode)
    parser.add_argument('--job', type=int, help='Job number for parallel processing')
    parser.add_argument('--total', type=int, help='Total number of jobs')
    
    # Input/Output paths
    parser.add_argument('--video_root', type=str, 
                       default='./data',
                       help='Root directory of videos')
    parser.add_argument('--output', type=str, default='./streamvggt',
                       help='Output directory for predictions/reconstructions')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints.pth',
                       help='Path to model checkpoint')
    
    # Processing parameters
    parser.add_argument('--fps_interval', type=float, default=1.0, 
                       help='Interval of frame sampling')
    parser.add_argument('--use_multi_gpu', action='store_true', default=True,
                       help='Use multi-GPU parallel processing (default: True)')
    parser.add_argument('--force_single_gpu', action='store_true', default=False,
                       help='Force single GPU processing even if multiple GPUs available')
    
    # Multi-method visualization parameters
    parser.add_argument('--ours_folder', type=str, default='reconstructions/ours',
                       help='Path to ours reconstruction folder')
    parser.add_argument('--baseline_folder', type=str, default='reconstructions/baseline',
                       help='Path to baseline reconstruction folder')
    parser.add_argument('--gt_folder', type=str, default='reconstructions/gt',
                       help='Path to ground truth reconstruction folder')
    parser.add_argument('--share', action='store_true', default=False,
                       help='Share the visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Initialize model based on mode (always use feature_only=False now since we handle it in inference)
    inferencer = StreamVGGTInference(args.checkpoint_path, feature_only=False, aux_feature = args.mode == 'export_memory')
    
    if args.mode == 'export_memory':
        run_export_memory(args, inferencer)
        
    elif args.mode == 'reconstruction':
        run_reconstruction(args, inferencer)
        
    elif args.mode == 'visualization':
        print("Launching Multi-Method 3D Reconstruction Comparison...")
        os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/tmp")
        
        # Set up folder paths
        base_folders = [args.ours_folder, args.baseline_folder, args.gt_folder]
        
        # Launch the multi-method visualization
        demo = gradio_multi_method_visualization(base_folders)
        demo.launch(
            share=args.share,
            inbrowser=True,
            server_name="0.0.0.0",
            server_port=7860
        )
    
    print("Process completed!")
import os
import cv2
import torch
import numpy as np
import sys
import tempfile
import shutil
from typing import Union, List, Dict, Optional
import glob

from .visual_util import predictions_to_glb
from .models.streamvggt import StreamVGGT
from .utils.load_fn import load_and_preprocess_images
from .utils.pose_enc import pose_encoding_to_extri_intri
from .utils.geometry import unproject_depth_map_to_point_map


class StreamVGGTInference:
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize StreamVGGT model for inference.
        
        Args:
            checkpoint_path: Path to local checkpoint. If None, downloads from HuggingFace.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU which may be slow")
        
        self.model = self._load_model(checkpoint_path)
        # Initialize past_key_values for continuous inference
        self.past_key_values = None
        self.past_key_values_camera = None
        self.total_processed_frames = 0
        
    def _load_model(self, checkpoint_path: Optional[str] = None) -> StreamVGGT:
        """Load and initialize the StreamVGGT model."""
        print("Initializing and loading StreamVGGT model...")
        
        model = StreamVGGT(feature_only=True)
        
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
    
    def reset_inference_state(self):
        """Reset the inference state for a new sequence."""
        self.past_key_values = None
        self.past_key_values_camera = None 
        self.total_processed_frames = 0
        print("StreamVGGT inference state reset.")
    
    def _initialize_past_key_values_if_needed(self):
        """Initialize past_key_values if they are None."""
        if self.past_key_values is None:
            self.past_key_values = [None] * self.model.aggregator.depth
        if self.past_key_values_camera is None:
            self.past_key_values_camera = [None] * self.model.camera_head.trunk_depth
        
    def _extract_frames_from_video(self, video_path: str, fps_interval: float = 0.25) -> List[np.ndarray]:
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
        frame_interval = int(fps * fps_interval) # 24*0.25=6
        
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
        
    def _process_input(self, input_data: Union[str, List[str], List[np.ndarray]], fps_interval: float) -> str:
        """
        Process input data and return path to temporary directory with images.
        
        Args:
            input_data: Can be:
                - String path to video file
                - List of string paths to image files  
                - List of numpy arrays (image frames)
                
        Returns:
            Path to temporary directory containing processed images
        """
        if isinstance(input_data, str): # Goes here
            # Single video file
            if input_data.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                frames = self._extract_frames_from_video(input_data, fps_interval) # all 360 frames are extracted, using fps_interval=0.25, so 360/(24*0.25)=60 frames
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
        

    def run_inference(self, input_data: Union[str, List[str], List[np.ndarray]], fps_interval: float) -> Dict:
        """
        Run StreamVGGT inference on input data.
        
        Args:
            input_data: Input video file path, list of image paths, or list of image arrays
            Currently is the video path
            
        Returns:
            Dictionary containing all predictions:
            - images: Input images tensor (S, 3, H, W)
            - world_points: 3D world points (S, H, W, 3)  
            - world_points_conf: Confidence for world points (S, H, W)
            - depth: Depth maps (S, H, W, 1)
            - depth_conf: Depth confidence (S, H, W)
            - pose_enc: Pose encoding (S, 9)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
            - intrinsic: Camera intrinsic matrices (S, 3, 3)
            - world_points_from_depth: World points from depth (S, H, W, 3)
        """
        # Process input to get temporary directory with images
        temp_dir = self._process_input(input_data, fps_interval)
        
        try:
            # Run model inference
            predictions = self._run_model_inference(temp_dir)
            return predictions
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    def _run_model_inference(self, target_dir: str) -> Dict:
        """Core model inference logic."""
        print(f"Processing images from {target_dir}")
        
        # Load and preprocess images
        image_names = glob.glob(os.path.join(target_dir, "images", "*"))
        image_names = sorted(image_names)
        print(f"Found {len(image_names)} images")
        
        if len(image_names) == 0:
            raise ValueError("No images found in input")
            
        images = load_and_preprocess_images(image_names).to(self.device) # [360, 3, 518, 518]
        print(f"Preprocessed images shape: {images.shape}") 
        
        # Prepare frames for model
        frames = []
        for i in range(images.shape[0]):
            image = images[i].unsqueeze(0)
            frame = {"img": image}
            frames.append(frame)
            
        # Run inference
        print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                output = self.model.export_memory(frames)
        
        return output

        
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
        """Save memory predictions to NPZ file."""
        np.save(save_path, predictions)
        # np.savez_compressed(save_path.replace('npy', 'npz'), arr=predictions)
    
    def run_continuous_inference(self, input_data: Union[str, List[str], List[np.ndarray]], fps_interval: float = 0.25) -> tuple:
        """
        Run StreamVGGT inference with continuous state preservation.
        
        Args:
            input_data: Input video file path, list of image paths, or list of image arrays
            fps_interval: Frame sampling interval in seconds
            
        Returns:
            Tuple of (global_memory, camera_data) where:
            - global_memory: All aggregated tokens concatenated [N, H, W, 1024]
            - camera_data: Dict containing last 4 frames camera info:
                * memory_key: Last 4 frames memory tokens [4, H, W, 1024]
                * extrinsic_key: Last 4 frames extrinsic matrices [4, 3, 4]
                * intrinsic_key: Last 4 frames intrinsic matrices [4, 3, 3]
        """
        # Process input to get temporary directory with images
        temp_dir = self._process_input(input_data, fps_interval)
        
        try:
            # Run model inference with continuous state
            global_memory, memory_key, extrinsic_key, intrinsic_key = self._run_continuous_model_inference(temp_dir)
            
            # for backward compatibility, return format is the same as AR, but the second return value contains extra information
            camera_data = {
                'memory_key': memory_key,
                'extrinsic_key': extrinsic_key, 
                'intrinsic_key': intrinsic_key
            }
            return global_memory, camera_data
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _run_continuous_model_inference(self, target_dir: str) -> tuple:
        """Core model inference logic with continuous state preservation."""
        print(f"Processing images from {target_dir}")
        
        # Load and preprocess images
        image_names = glob.glob(os.path.join(target_dir, "images", "*"))
        image_names = sorted(image_names)
        print(f"Found {len(image_names)} images")
        
        if len(image_names) == 0:
            raise ValueError("No images found in input")
            
        images = load_and_preprocess_images(image_names).to(self.device)
        print(f"Preprocessed images shape: {images.shape}") # Preprocessed images shape: torch.Size([20, 3, 280, 518])
        
        # Prepare frames for model
        frames = []
        for i in range(images.shape[0]):
            image = images[i].unsqueeze(0)
            frame = {"img": image}
            frames.append(frame)
        
        # Initialize past_key_values if needed
        self._initialize_past_key_values_if_needed()
        
        # Run inference with continuous state
        print("Running continuous inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        all_aggregated_tokens = {}
        all_camera_poses = {}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                for i, frame in enumerate(frames):
                    # Update frame index for continuous processing
                    frame_idx = self.total_processed_frames + i
                    
                    images_single = frame["img"].unsqueeze(0)  # [1, 1, C, H, W]
                    
                    # Run aggregator with past_key_values
                    aggregator_output = self.model.aggregator(
                        images_single, 
                        past_key_values=self.past_key_values,
                        use_cache=True, 
                        past_frame_idx=frame_idx
                    )
                    
                    if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                        aggregated_tokens, patch_start_idx, self.past_key_values = aggregator_output
                    else:
                        aggregated_tokens, patch_start_idx = aggregator_output
                    
                    # calculate camera pose (added for autoregressive camera parameter generation)
                    with torch.cuda.amp.autocast(enabled=False):
                        if self.model.camera_head is not None:
                            pose_enc, self.past_key_values_camera = self.model.camera_head(
                                aggregated_tokens, 
                                past_key_values_camera=self.past_key_values_camera, 
                                use_cache=True
                            )
                            pose_enc = pose_enc[-1]
                            camera_pose = pose_enc[:, 0, :].squeeze(0)  # remove batch dimension
                            all_camera_poses[frame_idx] = camera_pose.detach()
                    
                    # Extract features following export_memory logic
                    selected_tokens = [aggregated_tokens[idx] for idx in [4, 11, 17, 23]]
                    selected_tokens = torch.cat(selected_tokens, dim=1).detach().cpu()
                    
                    # Store the last 1024 dimensions
                    all_aggregated_tokens[frame_idx] = selected_tokens.detach()[:, :, :, -1024:]
        
        # Update total processed frames count
        self.total_processed_frames += len(frames)
        
        # Concatenate all tokens
        all_aggregated_tokens_cat = torch.cat(list(all_aggregated_tokens.values()), dim=0).cpu().numpy()
        
        # process camera parameters (take the last 4 frames as key)
        extrinsic_key = None
        intrinsic_key = None
        
        if all_camera_poses:
            # take the last 4 frames of camera poses
            recent_frame_indices = sorted(all_camera_poses.keys())[-4:]  # the last 4 frames
            recent_camera_poses = [all_camera_poses[idx] for idx in recent_frame_indices]
            recent_camera_poses_stacked = torch.stack(recent_camera_poses, dim=0)  # [4, 9]
            
            # get image size (assuming all frames have the same size)
            image_size = frames[0]["img"].shape[-2:]
            
            # import pose conversion function
            from .utils.pose_enc import pose_encoding_to_extri_intri
            
            # convert to extrinsic and intrinsic
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                recent_camera_poses_stacked.unsqueeze(0),  # [1, 4, 9]
                image_size
            )
            
            extrinsic_key = extrinsic.squeeze(0).cpu().numpy()  # [4, 3, 4]
            intrinsic_key = intrinsic.squeeze(0).cpu().numpy() if intrinsic is not None else None  # [4, 3, 3]
            
            print(f"Generated extrinsic_key shape: {extrinsic_key.shape}")
            print(f"Generated intrinsic_key shape: {intrinsic_key.shape if intrinsic_key is not None else None}")
        
        # take the last 4 frames of memory as memory_key
        memory_key = all_aggregated_tokens_cat[-4:] if len(all_aggregated_tokens_cat) >= 4 else all_aggregated_tokens_cat
        
        del all_aggregated_tokens
        del all_camera_poses
        
        print(f"Generated global memory shape: {all_aggregated_tokens_cat.shape}") # Generated global memory shape: (20, 4, 745, 1024)
        print(f"Generated memory_key shape: {memory_key.shape}")
        print(f"Total processed frames so far: {self.total_processed_frames}")
        
        return all_aggregated_tokens_cat, memory_key, extrinsic_key, intrinsic_key

# Example usage functions
def run_inference_on_video(video_path: str, checkpoint_path: Optional[str] = None) -> Dict:
    """
    Convenience function to run inference on a video file.
    
    Args:
        video_path: Path to input video file
        checkpoint_path: Optional path to model checkpoint
        
    Returns:
        Dictionary containing all predictions
    """
    inferencer = StreamVGGTInference(checkpoint_path)
    return inferencer.run_inference(video_path)
    # return inferencer.export_memory(video_path)


def run_inference_on_images(image_paths: List[str], checkpoint_path: Optional[str] = None) -> Dict:
    """
    Convenience function to run inference on a list of image files.
    
    Args:
        image_paths: List of paths to input image files
        checkpoint_path: Optional path to model checkpoint
        
    Returns:
        Dictionary containing all predictions
    """
    inferencer = StreamVGGTInference(checkpoint_path)
    return inferencer.run_inference(image_paths)


def run_inference_on_frames(frames: List[np.ndarray], checkpoint_path: Optional[str] = None) -> Dict:
    """
    Convenience function to run inference on a list of image frames.
    
    Args:
        frames: List of image frames as numpy arrays (BGR format)
        checkpoint_path: Optional path to model checkpoint
        
    Returns:
        Dictionary containing all predictions
    """
    inferencer = StreamVGGTInference(checkpoint_path)
    return inferencer.run_inference(frames)

def load_memory(path):
    memory = np.load(path, allow_pickle=True).item()
    
    for frame_id in memory:
        frame_memory = memory[frame_id]
        if 'aggregated_tokens_global' in frame_memory:
            global_memory = torch.tensor(frame_memory['aggregated_tokens_global']) # [1, 4, 782, 1024])
        else:
            global_memory = torch.tensor(frame_memory['aggregated_tokens'])[:, :, :, -1024:]
    
# Example usage
if __name__ == "__main__":
    from tqdm import tqdm
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=int, help='# function')
    parser.add_argument('--total', type=int, help='# total jobs')
    
    parser.add_argument('--video_root', type=str, default='./data/', help='Root directory of videos')
    parser.add_argument('--output', type=str, default='./data/streamvggt_predictions', help='Output directory for predictions')
    parser.add_argument('--fps_interval', type=float, default=0.25, help='Interval of frame sampling')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    
    video_root = args.video_root
    save_root = args.output
    inferencer = StreamVGGTInference()

    videos = sorted(os.listdir(video_root))
    videos = [video for video in videos if (not os.path.exists(f"{save_root}/{video.split('.')[0]}.npy")) and video.endswith('.mp4')]
    start_idx = args.job * len(videos) // args.total
    end_idx = (args.job + 1) * len(videos) // args.total
    
    videos = videos[start_idx:end_idx]
    for video in tqdm(videos):
        if os.path.exists(f"{save_root}/{video.split('.')[0]}.npy"):
            print(f"Skipping {video}, already processed.")
            continue
        video_path = os.path.join(video_root, video)
        try:
            global_memory, pts3d_feature = inferencer.run_inference(video_path, fps_interval=args.fps_interval)
            inferencer.save_predictions(global_memory, f"{save_root}/{video.split('.')[0]}.npy")
            del global_memory

            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing {video}: {e}")
            continue
        
        


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
        
    def _process_input(self, input_data: Union[str, List[str], List[np.ndarray]]) -> str:
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
        if isinstance(input_data, str):
            # Single video file
            if input_data.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                frames = self._extract_frames_from_video(input_data)
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
        

    def run_inference(self, input_data: Union[str, List[str], List[np.ndarray]]) -> Dict:
        """
        Run StreamVGGT inference on input data.
        
        Args:
            input_data: Input video file path, list of image paths, or list of image arrays
            
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
        temp_dir = self._process_input(input_data)
        
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
            
        images = load_and_preprocess_images(image_names).to(self.device)
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
                output = self.model.inference(frames)
                # output = self.model.export_memory(frames)
       
        # Extract results
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
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        return predictions
        
    def save_predictions(self, predictions: Dict, save_path: str):
        """Save predictions to NPZ file."""
        np.savez(save_path, **predictions)
        print(f"Predictions saved to {save_path}")
        
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


# Example usage
if __name__ == "__main__":
    # Example 1: Process video
    predictions = run_inference_on_video("/home/xwang378/scratch/2025/StreamVGGT/input_images_20250731_102143_185403/images/-0HwkO7TRmc.00.mp4")
    # Example 2: Process image files
    # image_files = ["img1.jpg", "img2.jpg", "img3.jpg"]
    # predictions = run_inference_on_images(image_files)
    
    # Example 3: Process numpy arrays
    # frames = [cv2.imread("img1.jpg"), cv2.imread("img2.jpg")]
    # predictions = run_inference_on_frames(frames)
    
    # Example 4: Full workflow with visualization
    inferencer = StreamVGGTInference()
    
    # Run inference (replace with your input)
    # predictions = inferencer.run_inference("path/to/input")
    
    # Save predictions
    # inferencer.save_predictions(predictions, "results.npz")
    
    # Create 3D visualization
    # inferencer.create_3d_visualization(predictions, "reconstruction.glb")
    
    print("StreamVGGT inference module ready!")
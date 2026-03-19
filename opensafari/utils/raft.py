import sys
sys.path.append('core')

import argparse, os, glob, json, logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import tqdm
import torch.multiprocessing as mp

from utils.core.raft import RAFT
from utils.core.utils.utils import InputPadder

# (Helper functions like load_image, compute_average_flow_magnitude, etc. are unchanged)
def load_image(imfile, device):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def compute_average_flow_magnitude(flow):
    u = flow[:, 0, :, :]; v = flow[:, 1, :, :]
    magnitude = torch.sqrt(u**2 + v**2)
    mean_motion = magnitude.mean(dim=(1,2))
    return mean_motion

class ImagePairDataset(Dataset):
    def __init__(self, root_path, video_folders):
        self.pairs = []
        self.folders = []
        for folder in tqdm.tqdm(video_folders, desc="Initializing Dataset", disable=dist.get_rank() != 0):
            full_folder_path = os.path.join(root_path, folder)
            if not os.path.isdir(full_folder_path):
                logging.warning(f"Folder {folder} not found in {root_path}, skipping.")
                continue
            images = sorted(glob.glob(os.path.join(full_folder_path, '*.jpg')))
            if len(images) < 2: continue
            for i in range(len(images) - 1):
                self.pairs.append((folder, images[i], images[i+1]))
            self.folders.append(folder)

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# --- DDP Worker Function (CORRECTED) ---
def ddp_demo(rank, world_size, frames_path, watermark_path, raft_path, args):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # --- MODEL INITIALIZATION: Must be done by ALL ranks unconditionally ---
    raw_state = torch.load(args.raft_model, map_location=device)
    new_state = OrderedDict((k.replace("module.", ""), v) for k, v in raw_state.items())
    model = RAFT(args).to(device)
    model.load_state_dict(new_state)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True) # RAFT might have unused params

    # --- DATA-DEPENDENT LOGIC ---
    os.makedirs(raft_path, exist_ok=True)
    raft_file_path = os.path.join(raft_path, 'raft.csv')
    already_processed_segs = set()
    if os.path.exists(raft_file_path):
        try:
            processed_df = pd.read_csv(raft_file_path)
            already_processed_segs = set(processed_df['video'])
        except Exception as e:
            if rank == 0: logging.warning(f"Could not read raft file. Error: {e}")

    try:
        df = pd.read_csv(watermark_path)
    except FileNotFoundError:
        if rank == 0: logging.error(f"Watermark file not found: {watermark_path}.")
        cleanup(); return

    eligible_videos = set(df[df['watermark_ratio'] < args.watermark_threshold]['video'])
    videos_to_process = sorted(list(eligible_videos - already_processed_segs))

    if rank == 0:
        logging.info(f"Found {len(eligible_videos)} eligible videos after watermark filter.")
        logging.info(f"After removing {len(already_processed_segs)} processed videos, {len(videos_to_process)} remain.")

    folder_motion, folder_counts = {}, {}
    if videos_to_process:
        dataset = ImagePairDataset(frames_path, videos_to_process)
        if len(dataset) > 0:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
            dataloader = DataLoader(
                dataset, batch_size=args.raft_batch_size, sampler=sampler,
                num_workers=args.workers, pin_memory=True, drop_last=False
            )
            if len(dataloader) > 0:
                model.eval()
                with torch.no_grad():
                    pbar = tqdm.tqdm(dataloader, desc=f"Rank {rank}", disable=rank!=0)
                    for folders, ims1, ims2 in pbar:
                        imgs1 = torch.cat([load_image(p, device) for p in ims1], dim=0)
                        imgs2 = torch.cat([load_image(p, device) for p in ims2], dim=0)
                        padder = InputPadder(imgs1.shape)
                        imgs1, imgs2 = padder.pad(imgs1, imgs2)
                        _, flow_up = model(imgs1, imgs2, iters=20, test_mode=True)
                        motions = compute_average_flow_magnitude(flow_up).cpu().numpy()
                        for f, m in zip(folders, motions):
                            folder_motion[f] = folder_motion.get(f, 0.0) + float(m)
                            folder_counts[f] = folder_counts.get(f, 0) + 1
            else:
                logging.info(f"[Rank {rank}] No videos assigned to this rank's dataloader.")
    elif rank == 0:
        logging.info("No new videos to process for RAFT. Skipping.")

    # --- SYNCHRONIZATION: ALL ranks must participate ---
    all_motions = [None] * world_size
    all_counts  = [None] * world_size
    dist.all_gather_object(all_motions, folder_motion)
    dist.all_gather_object(all_counts, folder_counts)

    if rank == 0:
        final_motion, final_counts = {}, {}
        for fm, fc in zip(all_motions, all_counts):
            for k, v in fm.items(): final_motion[k] = final_motion.get(k, 0.0) + v
            for k, v_c in fc.items(): final_counts[k] = final_counts.get(k, 0) + v_c
        
        newly_processed_avg_motion = {k: final_motion[k] / final_counts[k] for k, v in final_motion.items() if final_counts.get(k, 0) > 0}

        final_results = {}
        if already_processed_segs:
            try:
                processed_df = pd.read_csv(raft_file_path)
                final_results = pd.Series(processed_df.average_motion.values, index=processed_df.video).to_dict()
            except Exception as e:
                logging.warning(f"[Master] Could not re-read raft file. Error: {e}")

        final_results.update(newly_processed_avg_motion)
        
        if not final_results:
            logging.info("[Master] No new or old RAFT data to write.")
        else:
            with open(raft_file_path, 'w') as f:
                f.write('video,average_motion\n')
                for video, avg_motion in sorted(final_results.items()):
                    f.write(f'{video},{avg_motion:.6f}\n')
            logging.info(f"[Master] Saved RAFT scores for {len(final_results)} videos -> {raft_file_path}")

    cleanup()

def raft(frames_path, watermark_path, raft_path, args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    watermark_file =  os.path.join(watermark_path, 'watermark.csv')
    mp.spawn(ddp_demo,
             args=(args.world_size, frames_path, watermark_file, raft_path, args),
             nprocs=args.world_size,
             join=True)
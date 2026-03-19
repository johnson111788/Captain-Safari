#!/usr/bin/env python
"""
Distributed watermark detection (DDP) with *global* frame batching,
refactored for direct Python invocation instead of torchrun.
DDP setup style is aligned with raft.py.
"""

from __future__ import annotations

import os, csv, logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Set
from argparse import Namespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipForImageClassification

# (Dataset, collate_fn, setup, cleanup, build_model, preds_for_batch functions are unchanged)
class FrameDataset(Dataset):
    def __init__(self, records: List[Tuple[str, Path]]):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        seg, path = self.records[idx]
        return seg, Image.open(path).convert("RGB")

def collate_fn(batch):
    segs, imgs = zip(*batch)
    return list(segs), list(imgs)

def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def build_model(device: torch.device, watermark_model: str):
    m = SiglipForImageClassification.from_pretrained(watermark_model).to(device).eval()
    return m

@torch.inference_mode()
def preds_for_batch(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    logits = model(**inputs).logits
    return torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu()


# -----------------------------------------------------------------------------
# DDP Worker Function (CORRECTED) ─────────────────────────────────────────────
# -----------------------------------------------------------------------------
def run_worker(rank: int, world_size: int, segs_to_process: List[str], frame_dir: Path, output_file: Path, args: Namespace, already_processed_segs: Set[str]):
    """The main function executed by each DDP process."""
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # --- MODEL INITIALIZATION: Must be done by ALL ranks unconditionally ---
    model = build_model(device, args.watermark_model)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None, find_unused_parameters=False)
    
    # --- DATA-DEPENDENT LOGIC ---
    seg_shard = segs_to_process[rank::world_size]
    records: List[Tuple[str, Path]] = []
    for vid in seg_shard:
        frame_path = frame_dir / vid
        if frame_path.is_dir():
            records.extend((vid, p) for p in sorted(frame_path.glob("frame_*.jpg")))

    ratios_local = {}
    if records:
        logging.info(f"[Rank {rank}] Processing {len(seg_shard)} videos with {len(records)} total frames.")
        dataset = FrameDataset(records)
        loader = DataLoader(
            dataset, batch_size=args.watermark_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
        )
        processor = AutoImageProcessor.from_pretrained(args.watermark_model)

        total_counts = defaultdict(int)
        wm_counts = defaultdict(int)

        pbar = tqdm(loader, desc=f"Rank {rank} batches", disable=rank != 0, position=rank)
        for vids, imgs in pbar:
            preds = preds_for_batch(model.module, processor, imgs, device)
            for vid, pred in zip(vids, preds.tolist()):
                total_counts[vid] += 1
                if pred == args.watermark_index:
                    wm_counts[vid] += 1
        
        ratios_local = {
            vid: wm_counts[vid] / total_counts[vid] for vid in total_counts
        }
    else:
        logging.info(f"[Rank {rank}] No new frames assigned to this rank.")

    # --- SYNCHRONIZATION: ALL ranks must participate ---
    gathered = [None] * world_size
    dist.all_gather_object(gathered, ratios_local)

    if rank == 0:
        newly_processed_ratios = {}
        for d in gathered:
            if d: newly_processed_ratios.update(d)
        
        final_ratios = {}
        if output_file.exists() and already_processed_segs:
            try:
                with output_file.open('r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['video'] in already_processed_segs:
                            final_ratios[row['video']] = float(row['watermark_ratio'])
            except Exception as e:
                logging.warning(f"\n[Master] Could not read existing data from {output_file}. Error: {e}")
        
        final_ratios.update(newly_processed_ratios)

        if not final_ratios:
             logging.info("\n[Master] No new or old data to write.")
        else:
            with output_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["video", "watermark_ratio"])
                for vid, ratio in sorted(final_ratios.items()):
                    writer.writerow([vid, f"{ratio:.6f}"])
            logging.info(f"\n[Master] Saved watermark ratios for {len(final_ratios)} videos -> {output_file}")

    cleanup()

# (The `watermark` launcher function is unchanged from the previous version)
def watermark(
    frame_dir: str,
    output_path: str,
    args,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    frame_dir = Path(frame_dir)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "watermark.csv"

    args_ns = Namespace(
        watermark_batch_size=args.watermark_batch_size,
        workers=args.workers,
        watermark_model=args.watermark_model,
        watermark_threshold=args.watermark_threshold,
        watermark_index=args.watermark_index,
        world_size=args.world_size
    )

    try:
        all_available_segs = sorted(os.listdir(frame_dir))
        if not all_available_segs:
            logging.error("No video directories found in 'frame_dir'.")
            return
    except FileNotFoundError:
        logging.error(f"'frame_dir' not found: {frame_dir}")
        return

    already_processed_segs = set()
    if output_file.exists():
        try:
            with output_file.open('r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if row: already_processed_segs.add(row[0])
            logging.info(f"Found {len(already_processed_segs)} already processed videos in {output_file}.")
        except Exception as e:
            logging.warning(f"Could not read existing watermark file, will re-process all. Error: {e}")
            already_processed_segs = set()

    segs_to_process = sorted(list(set(all_available_segs) - already_processed_segs))

    if not segs_to_process:
        logging.info("All videos have already been processed for watermarks. Skipping.")
        return

    logging.info(f"Found {len(all_available_segs)} total videos. Launching {args_ns.world_size} GPU processes for {len(segs_to_process)} new videos...")

    mp.spawn(
        run_worker,
        args=(args_ns.world_size, segs_to_process, frame_dir, output_file, args_ns, already_processed_segs),
        nprocs=args_ns.world_size,
        join=True,
    )
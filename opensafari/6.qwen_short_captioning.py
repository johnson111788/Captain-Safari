import os
import csv
import math
import argparse
import logging
import multiprocessing as mp
from queue import Empty
from typing import List, Tuple, Dict, Set

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# -------------------------------
# Prompt (kept concise & deterministic-enough)
# -------------------------------
PROMPT = (
    """
Generate a brief, one-sentence caption that captures the essence of this video.
Focus on the camera movements, scene, or subject matter. Keep it concise and descriptive.
Avoid mentioning technical details like drone presence or fine details.
Output only the final caption with no additional commentary.
"""
).strip()


# -------------------------------
# Helpers
# -------------------------------

def list_videos(directory_path: str) -> List[str]:
    VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv")
    videos = []
    for entry in os.scandir(directory_path):
        if entry.is_file() and entry.name.lower().endswith(VIDEO_EXTENSIONS):
            videos.append(os.path.abspath(entry.path))
    return videos


def read_existing_basenames(output_file: str) -> Set[str]:
    """Return a set of basenames that have already been written to output_file."""
    done: Set[str] = set()
    if not os.path.exists(output_file):
        return done
    try:
        with open(output_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header if present
            for row in reader:
                if len(row) >= 1 and row[0]:
                    done.add(os.path.basename(row[0]))
    except Exception as e:
        logging.warning(f"Failed to read existing results from {output_file}: {e}")
    return done


def ensure_header(output_file: str):
    """Ensure CSV has a header; create file if missing/empty."""
    needs_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    if needs_header:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "short_caption"])


def split_videos_for_gpus(video_files: List[str], num_gpus: int) -> List[List[str]]:
    if not video_files:
        return []
    num_gpus = max(1, num_gpus)
    per = math.ceil(len(video_files) / num_gpus)
    batches = []
    for i in range(num_gpus):
        s, e = i * per, min((i + 1) * per, len(video_files))
        if s < len(video_files):
            batches.append(video_files[s:e])
            logging.info(f"GPU {i}: assigned {len(batches[-1])} videos (indices {s}-{e-1})")
    return batches


# -------------------------------
# Worker
# -------------------------------

def _mute_third_party_logs():
    # Optional: silence noisy libs (torchvision video probe, etc.)
    for name in ["torchvision", "torchvision.io", "torchvision.io.video"]:
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.propagate = False
    class _DropTorchVision(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "torchvision:" not in msg
    logging.getLogger().addFilter(_DropTorchVision())


def _build_batch_inputs(processor, device: str, video_paths: List[str]):
    """
    Build batched inputs while handling the case where a modality is entirely absent.
    Qwen processors expect `images=None` if no images exist, not `[None, None, ...]`.
    Similarly for `videos`.
    """
    texts = []
    images_list = []
    videos_list = []

    any_image = False
    any_video = False

    for vp in video_paths:
        msgs = [{
            "role": "user",
            "content": [
                {"type": "video", "video": vp},
                {"type": "text", "text": PROMPT},
            ],
        }]
        chat_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_inp, vid_inp = process_vision_info(msgs)

        texts.append(chat_text)
        images_list.append(img_inp)
        videos_list.append(vid_inp)

        if img_inp is not None:
            any_image = True
        if vid_inp is not None:
            any_video = True

    # Normalize None lists to what the processor expects
    images_arg = None if not any_image else [x if x is not None else [] for x in images_list]
    videos_arg = None if not any_video else [x if x is not None else [] for x in videos_list]

    inputs = processor(
        text=texts,
        images=images_arg,
        videos=videos_arg,
        padding=True,
        return_tensors="pt",
    ).to(device)
    return inputs


def _generate_batch(model, processor, inputs):
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    outs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [o.strip() for o in outs]


def gpu_worker_process(
    gpu_id: int,
    video_batch: List[str],
    results_queue: mp.Queue,
    model_name: str,
    min_pixels: int,
    max_pixels: int,
    batch_size: int,
):
    """Each GPU loads its own model once and streams per-video results back.

    Messages sent to results_queue:
      ("result", gpu_id, video_path, caption)
      ("done", gpu_id, processed_count)
    """
    import torch

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - [GPU %(process)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _mute_third_party_logs()

    try:
        if not torch.cuda.is_available():
            results_queue.put(("result", gpu_id, f"ERROR_GPU_{gpu_id}", "ERROR: CUDA not available"))
            results_queue.put(("done", gpu_id, 0))
            return

        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        logging.info(f"GPU {gpu_id} worker started, {len(video_batch)} videos, batch_size={batch_size}")

        # Load model & processor ON this GPU
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        logging.info(f"Model loaded on {device}")

        processed = 0
        bs = max(1, int(batch_size))
        for s in range(0, len(video_batch), bs):
            sub_paths = video_batch[s : s + bs]
            try:
                inputs = _build_batch_inputs(processor, device, sub_paths)
                with torch.no_grad():
                    captions = _generate_batch(model, processor, inputs)
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"GPU {gpu_id} OOM on batch of {len(sub_paths)}; falling back to single-item loop: {e}")
                torch.cuda.empty_cache()
                captions = []
                for vp in sub_paths:
                    try:
                        si_inputs = _build_batch_inputs(processor, device, [vp])
                        with torch.no_grad():
                            si_caps = _generate_batch(model, processor, si_inputs)
                        captions.append(si_caps[0] if si_caps else "")
                    except Exception as ie:
                        captions.append(f"ERROR: {str(ie)}")
                    finally:
                        try:
                            del si_inputs
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
            except Exception as e:
                # If whole batch fails for non-OOM reason, mark each item with error and continue
                captions = [f"ERROR: {str(e)}" for _ in sub_paths]

            # Stream results
            for vp, cap in zip(sub_paths, captions):
                try:
                    results_queue.put(("result", gpu_id, vp, cap))
                except Exception as e:
                    logging.error(f"Queue put failed on GPU {gpu_id} for {os.path.basename(vp)}: {e}")
                processed += 1

            try:
                del inputs
            except Exception:
                pass
            torch.cuda.empty_cache()

        results_queue.put(("done", gpu_id, processed))

    except Exception as e:
        try:
            results_queue.put(("result", gpu_id, f"ERROR_GPU_{gpu_id}", f"ERROR: {str(e)}"))
        finally:
            results_queue.put(("done", gpu_id, 0))


# -------------------------------
# Main (multi-GPU only, stream-writing)
# -------------------------------

def process_videos_multi_gpu_stream(
    directory_path: str,
    output_file: str = "short_captions.csv",
    num_gpus: int = 8,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
    batch_size: int = 4,
):
    logging.info(f"Scanning for videos in '{directory_path}'...")
    all_videos = list_videos(directory_path)
    if not all_videos:
        logging.info("No video files found. Exiting.")
        return

    logging.info(f"Found {len(all_videos)} total video files.")

    # Resume support: skip already processed basenames
    done_basenames = read_existing_basenames(output_file)
    remaining = [v for v in all_videos if os.path.basename(v) not in done_basenames]
    if not remaining:
        logging.info("All videos already processed. Nothing to do.")
        return

    available_gpus = torch.cuda.device_count()
    if available_gpus <= 0:
        logging.error("No GPUs available.")
        return

    num_gpus = max(1, min(num_gpus, available_gpus))
    logging.info(f"Using {num_gpus} / {available_gpus} GPUs.")

    batches = split_videos_for_gpus(remaining, num_gpus)
    if not batches:
        logging.info("No batches generated. Exiting.")
        return

    ensure_header(output_file)

    # Start workers
    ctx = mp.get_context("spawn")
    results_queue: mp.Queue = ctx.Queue(maxsize=num_gpus * 4)

    procs: List[mp.Process] = []
    for gid, batch in enumerate(batches):
        if not batch:
            continue
        p = ctx.Process(
            target=gpu_worker_process,
            args=(gid, batch, results_queue, model_name, min_pixels, max_pixels, batch_size),
            daemon=True,
        )
        p.start()
        procs.append(p)
        logging.info(f"Spawned GPU {gid} worker with {len(batch)} videos")

    # Stream-write results as they arrive
    total_expected = len(remaining)
    done_workers: Set[int] = set()
    written_basenames: Set[str] = set(done_basenames)  # prevent duplicates in the CSV

    with open(output_file, "a", newline="", encoding="utf-8") as f, \
         tqdm(total=total_expected, desc="Processing Videos (Multi-GPU)", unit="vid", dynamic_ncols=True) as pbar:
        writer = csv.writer(f)

        while len(done_workers) < len(procs):
            try:
                msg = results_queue.get(timeout=600)  # 10 min timeout per message allows for model load & long videos
            except Empty:
                # Check liveness; if all dead, break to avoid hang
                alive = [p.is_alive() for p in procs]
                logging.warning(f"Queue wait timeout. Alive workers: {sum(alive)}/{len(procs)}")
                if not any(alive):
                    break
                else:
                    continue

            tag = msg[0]
            if tag == "result":
                _, gid, video_path, caption = msg
                base = os.path.basename(video_path)
                if base not in written_basenames:
                    writer.writerow([base, caption])
                    f.flush()
                    os.fsync(f.fileno())
                    written_basenames.add(base)
                    pbar.update(1)
                    pbar.set_postfix({"GPU": gid, "last": base[:48]})
            elif tag == "done":
                _, gid, count = msg
                done_workers.add(gid)
                logging.info(f"GPU {gid} reports done: {count} processed")
            else:
                logging.warning(f"Unknown message tag: {tag}")

    # Join & cleanup
    for i, p in enumerate(procs):
        try:
            p.join(timeout=60)
            if p.is_alive():
                logging.warning(f"Process {i} still alive after timeout, terminating...")
                p.terminate()
                p.join(timeout=30)
                if p.is_alive():
                    logging.error(f"Process {i} still alive after terminate, killing...")
                    p.kill()
        except Exception as e:
            logging.error(f"Error joining process {i}: {e}")

    logging.info(f"Multi-GPU stream-write completed. Output: {output_file}")


# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    mp.set_start_method("spawn", force=True)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler("short_caption.log"), logging.StreamHandler()],
    )
    _mute_third_party_logs()

    parser = argparse.ArgumentParser(
        description="Generate short captions for videos using Qwen2.5-VL-7B model (multi-GPU, stream-writing, mini-batch)."
    )
    parser.add_argument("--directory", default="./data-slices", help="Directory with video files.")
    parser.add_argument("--output", "-o", default="short_captions.csv", help="Output CSV path.")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model id or local path.")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28, help="Processor min_pixels.")
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28, help="Processor max_pixels.")
    parser.add_argument("--batch-size", type=int, default=4, help="Generation mini-batch size per GPU worker.")

    args = parser.parse_args()

    available = torch.cuda.device_count()
    logging.info(f"Available GPUs: {available}")
    if available <= 0:
        logging.error("CUDA not available. Exiting.")
    else:
        process_videos_multi_gpu_stream(
            directory_path=args.directory,
            output_file=args.output,
            num_gpus=args.num_gpus,
            model_name=args.model,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            batch_size=args.batch_size,
        )

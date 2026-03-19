#!/bin/bash

# Multi-GPU parallel pre-processing input_image script
# Usage: bash utils/run_pretokenize_image_multi_gpu.sh

# ==================== Configuration parameters ====================

# GPU configuration
NUM_GPUS=8  # Number of GPUs used
GPU_IDS=(0 1 2 3 4 5 6 7)  # List of GPU IDs to use

# Dataset configuration
DATASET_BASE_PATH="./data"
DATASET_METADATA_PATH="./data/metadata.csv"
OUTPUT_DIR="./data"

# Model configuration (only VAE)
MODEL_ID_WITH_ORIGIN_PATHS="PAI/Wan2.2-Fun-5B-Control-Camera:Wan2.2_VAE.pth"

# Video parameters
NUM_FRAMES=121
HEIGHT=704
WIDTH=1280

# VAE parameters
TILED=""  # If using tiled encoding, set to "--tiled"
TILE_SIZE_H=34
TILE_SIZE_W=34
TILE_STRIDE_H=18
TILE_STRIDE_W=16

# ==================== Execution ====================

echo "========================================"
echo "Multi-GPU parallel pre-processing input_image"
echo "========================================"
echo "Number of GPUs used: ${NUM_GPUS}"
echo "GPU IDs: ${GPU_IDS[@]}"
echo "Dataset: ${DATASET_BASE_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "========================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Start a process on each GPU
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=${GPU_IDS[$i]}
    
    echo "Starting GPU ${GPU_ID} (process $i/${NUM_GPUS})..."
    
    # Use CUDA_VISIBLE_DEVICES to limit each process to only see one GPU
    CUDA_VISIBLE_DEVICES=${GPU_ID} python utils/pretokenize_image.py \
        --dataset_base_path "${DATASET_BASE_PATH}" \
        --dataset_metadata_path "${DATASET_METADATA_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}" \
        --num_frames ${NUM_FRAMES} \
        --height ${HEIGHT} \
        --width ${WIDTH} \
        ${TILED} \
        --tile_size_h ${TILE_SIZE_H} \
        --tile_size_w ${TILE_SIZE_W} \
        --tile_stride_h ${TILE_STRIDE_H} \
        --tile_stride_w ${TILE_STRIDE_W} \
        --gpu_id $i \
        --num_gpus ${NUM_GPUS} \
        > "${OUTPUT_DIR}/pretokenize_image_gpu${i}.log" 2>&1 &
    
    # Save process ID
    PIDS[$i]=$!
    
    # Slightly delay to avoid memory peak due to simultaneous model loading
    sleep 5
done

echo ""
echo "All GPU processes have been started!"
echo "Process IDs: ${PIDS[@]}"
echo ""
echo "Monitoring log files:"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  GPU ${GPU_IDS[$i]}: ${OUTPUT_DIR}/pretokenize_image_gpu${i}.log"
done
echo ""
echo "Waiting for all processes to complete..."
echo ""

# Wait for all background processes to complete
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PID=${PIDS[$i]}
    wait $PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ GPU ${GPU_IDS[$i]} (PID: ${PID}) completed"
    else
        echo "✗ GPU ${GPU_IDS[$i]} (PID: ${PID}) failed (exit code: ${EXIT_CODE})"
    fi
done

# Calculate total elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================"
echo "All GPUs processing completed!"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "========================================"
echo ""
echo "Output location:"
echo "  - Image Y latents: ${OUTPUT_DIR}/image_y_latents/"
echo "  - Image single latents: ${OUTPUT_DIR}/image_single_latents/"
echo "  - Metadata: ${OUTPUT_DIR}/*.encoded_with_image.csv"
echo ""

# Statistics
TOTAL_Y_FILES=$(find "${OUTPUT_DIR}/image_y_latents/" -name "*.pt" 2>/dev/null | wc -l)
TOTAL_SINGLE_FILES=$(find "${OUTPUT_DIR}/image_single_latents/" -name "*.pt" 2>/dev/null | wc -l)

echo "Statistics:"
echo "  - Image Y latents: ${TOTAL_Y_FILES} files"
echo "  - Image single latents: ${TOTAL_SINGLE_FILES} files"
echo ""

if [ ${TOTAL_Y_FILES} -eq ${TOTAL_SINGLE_FILES} ]; then
    echo "✓ All files have the same number of files, pre-processing successful!"
else
    echo "⚠ Files have different numbers, please check the log files"
fi


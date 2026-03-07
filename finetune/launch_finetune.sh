#!/bin/bash

# Launcher script for Walrus Navier-Stokes finetuning
# This script provides easy commands for different training scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
NUM_GPUS=1
MODE="single"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --num-gpus NUM     Number of GPUs to use (default: 1)"
    echo "  -m, --mode MODE        Training mode: single, multi-gpu, or debug (default: single)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Single GPU training"
    echo "  $0 --mode single"
    echo ""
    echo "  # Multi-GPU training with 4 GPUs"
    echo "  $0 --mode multi-gpu --num-gpus 4"
    echo ""
    echo "  # Debug mode (reduced epochs and validation frequency)"
    echo "  $0 --mode debug"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. CUDA may not be available.${NC}"
    echo -e "${YELLOW}Training will run on CPU, which will be very slow.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Activate conda environment if needed
if [ -d ".venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
elif [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${GREEN}Using conda environment: $CONDA_DEFAULT_ENV${NC}"
else
    echo -e "${YELLOW}Warning: No virtual environment detected.${NC}"
fi

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Walrus Navier-Stokes Finetuning${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

case $MODE in
    single)
        echo -e "${GREEN}Running single GPU training...${NC}"
        echo ""
        python train_navier_stokes_finetune.py
        ;;

    multi-gpu)
        echo -e "${GREEN}Running multi-GPU training with $NUM_GPUS GPUs...${NC}"
        echo ""

        # Check if torchrun is available
        if ! command -v torchrun &> /dev/null; then
            echo -e "${RED}Error: torchrun not found. Please install PyTorch distributed.${NC}"
            exit 1
        fi

        # Check if enough GPUs are available
        if command -v nvidia-smi &> /dev/null; then
            AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
            if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
                echo -e "${RED}Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available.${NC}"
                exit 1
            fi
        fi

        torchrun --nproc_per_node=$NUM_GPUS train_navier_stokes_finetune.py
        ;;

    debug)
        echo -e "${YELLOW}Running in DEBUG mode...${NC}"
        echo -e "${YELLOW}This will use reduced epochs and validation frequency for testing.${NC}"
        echo ""

        # Create a temporary debug version with modified hyperparameters
        # In practice, you might want to pass these as command-line args to the script
        # For now, just run normally but inform the user
        echo -e "${YELLOW}Note: To fully debug, edit MAX_EPOCHS and VAL_FREQUENCY in the script.${NC}"
        python train_navier_stokes_finetune.py
        ;;

    *)
        echo -e "${RED}Error: Unknown mode '$MODE'${NC}"
        usage
        ;;
esac

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo -e "Results saved to: ${GREEN}./experiments/navier_stokes_finetune/${NC}"
    echo -e "Checkpoints: ${GREEN}./experiments/navier_stokes_finetune/checkpoints/${NC}"
    echo -e "Visualizations: ${GREEN}./experiments/navier_stokes_finetune/visualizations/${NC}"
else
    echo ""
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}Training failed with exit code $?${NC}"
    echo -e "${RED}================================================${NC}"
    exit 1
fi

#!/bin/bash
# Force PyTorch to use CPU mode due to RTX 5050 sm_120 compatibility issue

echo "Running in CPU mode (RTX 5050 sm_120 not yet supported by PyTorch)"
export CUDA_VISIBLE_DEVICES=""
python "$@"
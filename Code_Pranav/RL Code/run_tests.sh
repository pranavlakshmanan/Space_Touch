#!/bin/bash

# SC-1 Model Testing Script
# Automatically finds latest model and runs comprehensive tests

echo "🚀 SC-1 Model Testing Suite"
echo "=============================="

# Activate conda environment
source /home/pralak/miniconda3/etc/profile.d/conda.sh
conda activate space_touch

# Navigate to correct directory
cd "/home/pralak/Space_Touch/Code_Pranav/RL Code"

# Check if we want visualization (default: yes)
if [ "$1" == "--no-vis" ]; then
    echo "📺 Visualization: DISABLED"
    python test_latest_model.py --no-vis --episodes 5
else
    echo "📺 Visualization: ENABLED"
    echo "💡 Use './run_tests.sh --no-vis' to disable visualization"
    echo ""
    python test_latest_model.py --episodes 5
fi

echo ""
echo "✅ Testing completed!"
echo "📊 Check ./SC1_Model_Tests/ for detailed results and plots"
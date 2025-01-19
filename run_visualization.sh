#!/bin/bash

# Set common parameters
PYTHONPATH=$PYTHONPATH:.
IMAGE_SIZE="100 75"
FPS=5

# List of objects to visualize
OBJECTS=(
    "chair"
    "drums" 
    "ficus"
    "hotdog"
    "lego"
    "materials"
    "mic"
    "ship"
)

# Create visualizations for each object
for obj in "${OBJECTS[@]}"; do
    echo "Creating visualization for $obj..."
    
    # Create output directory
    OUTPUT_DIR="visualizations/$obj"
    mkdir -p "$OUTPUT_DIR"
    
    # Run visualization
    PYTHONPATH=$PYTHONPATH:. python utils/visualization.py \
        --checkpoint "checkpoints/$obj/best_model.pt" \
        --output-dir "$OUTPUT_DIR" \
        --image-size $IMAGE_SIZE \
        --fps $FPS
        
    echo "Completed visualization for $obj"
    echo "----------------------------------------"
done

echo "All visualizations complete!"

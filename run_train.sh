#!/bin/bash

# List of datasets
DATASETS=("Chair" "Drums" "Ficus" "Hotdog" "Lego" "Materials" "Mic" "Ship")
CONFIG_FILE="configs/default_config.yaml"

# Function to update dataset path in config
update_dataset_path() {
    local dataset=$1
    # Use sed to replace the dataset path line
    # The -i flag means edit in place, the backup extension is ''
    sed -i'' -e "s|data:.*|dataset_path: \"data/nerf_synthetic/Synthetic_NeRF/${dataset}/processed\"|" $CONFIG_FILE
    # Update wandb name
    sed -i'' -e "s|wandb_name:.*|wandb_name: \"${dataset,,}\"|" $CONFIG_FILE  # ${dataset,,} converts to lowercase
}

# Main execution loop
for dataset in "${DATASETS[@]}"; do
    echo "======================================"
    echo "Processing dataset: $dataset"
    echo "======================================"
    
    # Update config file
    update_dataset_path "$dataset"
    
    # Run training script
    # Replace with your actual training command
    python main.py train --config $CONFIG_FILE --gpu 0,1,2
    
    # Optional: add a small delay between runs
    sleep 30
done
#!/bin/zsh

# Declare an array of dataset paths
datasets=(
    "gs://gresearch/android-in-the-wild/general"
)

# Create a local directory to store the datasets
mkdir -p datasets

# Loop through each dataset and download it
for dataset in "${datasets[@]}"; do
    echo "Downloading $dataset..."
    gsutil -m -o "GSUtil:parallel_process_count=1" cp -r "$dataset" ./datasets/
done

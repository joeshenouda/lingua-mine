## Creates the validation dataset from the chunked jsonl files for fineweb-edu 100b
import argparse
import os
import time
import subprocess
import requests
from huggingface_hub import snapshot_download

path_fineweb_edu_100b = "/scratch/scratch/jsheno/pnorms_lingua/lingua/data/fineweb_edu_100bt"

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

suffix = ".jsonl"
dataset = "fineweb_edu_100bt"
prefix = f"{dataset}.chunk."

k_validation = 10000  # Number of lines to take from each chunk for validation


# Create validation set and remove lines from chunks
validation_file = f"{path_fineweb_edu_100b}/{dataset}.val{suffix}"
for i in range(64):
    chunk_file = f"{path_fineweb_edu_100b}/{prefix}{i:02d}{suffix}"
    
    # Puts the first k_validation lines of each chunk file to the validation file
    run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")

    # Deletes the first k_validation lines from the chunk file
    run_command(f"sed -i '1,{k_validation}d' {chunk_file}")
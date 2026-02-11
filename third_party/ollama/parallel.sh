#!/bin/bash

HOST_NAME=$(hostname)
HOST=${HOST_NAME}.snu.vision

# Define the base port number
BASE_PORT=15151

# Number of server instances
NUM_SERVERS=8

# Configuration
MODELS=~/.ollama/models
KEEP_ALIVE=10m
NUM_PARALLEL=16
FLASH_ATTENTION=1
MAX_LOADED_MODELS=1

# Loop to start each server instance in the background
for ((i=0; i<NUM_SERVERS; i++)); do
   PORT=$((BASE_PORT + i))
   CUDA_VISIBLE_DEVICES=$i OLLAMA_HOST=${HOST}:${PORT} OLLAMA_MODELS=${MODELS} OLLAMA_KEEP_ALIVE=${KEEP_ALIVE} OLLAMA_NUM_PARALLEL=${NUM_PARALLEL} OLLAMA_FLASH_ATTENTION=${FLASH_ATTENTION} OLLAMA_MAX_LOADED_MODELS=${MAX_LOADED_MODELS} ollama serve &
   echo "Started ${i} server instance on ${HOST}:${PORT}"
   sleep 1
done

echo "All server instances started."
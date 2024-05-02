#!/bin/bash

########################################################################################
# Note that you need to set up conda environment for training and evaluation separately
# Training environment is for training the Retroformer model
# Evaluation environment is for evaluating the model on downstream tasks
# Trained model is served using FastChat server
########################################################################################

## TRAINING ENVIRONMENT ##
conda create -n train python=3.10 -y
pip install -r requirements.txt

## EVALUATION ENVIRONMENT ##
# Step 1. Run FastChat server
# Go to llm/serve.sh and run the command one by one

# Step 2. Install THREE separate Python environments for evaluation
# 1. HotPotQA
# conda create -n hotpotqa python=3.10 -y
# pip install -r experiments/hotpotqa_runs/requiresments.txt
# 2. Webshop
# Install the requirements for the webshop
# https://github.com/princeton-nlp/WebShop
# 3. Alfworld
# Install the requirements for the Alfworld
# https://github.com/alfworld/alfworld

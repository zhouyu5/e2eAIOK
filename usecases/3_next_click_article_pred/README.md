# Next Clicked Article Prediction

## Problem Description
[https://www.recsyschallenge.com/2024/](https://www.recsyschallenge.com/2024/)
A problem aims to predict which article a user will click from a list of articles that was seen during a specific impression. Utilizing the user's browsing history, session details (like time and device used), and personal metadata (including gender and age), along with a list of candidate news articles, listed in an impression log. 

The objective is to rank the candidate articles based on the user's personal preferences. This involves developing models that encapsulate both the users and the articles through their content and the users' interests. The models are to estimate the likelihood of a user clicking each article by evaluating the compatibility between the article's content and the user's preferences. The articles are ranked based on these likelihood scores, and the precision of these rankings is measured against the actual selections made by users.

## Solution Overview
Under Developmemt...

## Getting Started

### Step 1: Prepare the dataset
- Download the RecSys Challenge 2024 dataset from the official [website](https://recsys.eb.dk/#dataset-container)

### Step 2: Prepare the environment
```bash
export WORKDIR=`pwd`
export USECASE_PATH=${WORKDIR}"/usecases/3_next_click_article_pred/"
mkdir -p ${USECASE_PATH}/dataset;
# copy train and test folders here
```

### Step 3: Training the model
- Trigger training using the following command
```bash
cd ${WORKDIR}/recsys-kits/models/recsys24/docker
docker-compose up recsys24-train
```

### Step 4: Update trained model to model repository 
- Renamed the trained model into `model.txt`, and move it to the triton models server
```bash
cp ${USECASE_PATH}/output/models/*.chk ${USECASE_PATH}/model_store/prediction/1/model.txt
```

## Step 5: launch serving
- Start Triton Inference Server
```bash
cd ${WORKDIR}/recsys-kits/serve/triton_serve/docker
docker-compose up triton-server-cpu
```
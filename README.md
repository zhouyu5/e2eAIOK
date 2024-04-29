# RecSys Reference Kit
Offering leading RecSys development tools/models and use cases, aiming to boost productivity/performance in creating recsys solution. It embrace Hugging Face community and fully compatible with Hugging Face Library.

## Quick Link

* [recsys-kits](recsys-kits): RecSys Development Kit, including models with train / inference apis, and serving options.
* [usecases](usecases): RecSys usecases, including various tasks, from CTR prediction to session-based recommendation, etc. It's based on popular academic datasets or RecSys Challenge.

## Key Features
* **High maintainability, Hugging Face compatible**: we do not reinvent the wheel, so we fully embrace the open-source community, especially the Hugging Face community, including:
    * 🤗 `transformers` library: We use it to build/incoporate transformer based models, launch training, etc.
    * 🤗 `datasets` library: We use it to load/preprocess dataset, share dataset (in progress).
    * 🤗 `Hub` library: We use it to share models/demos (in progress).
    * 🤗 `evaluate` library: We use it to evaluate models and datasets.
    * 🤗 `optimum` library: We use it to run models on targeted hardware with maximum efficiency, such as intel Gaudi accelerator.
    * 🤗 `accelerate` library: We use it to support distributed training (in progress).
* **Easy to use, Low code**: users could launch training by merely declared 3 configs without writing one line code, which are:
    * `config.json`: it's model releated configuration, such as model layers, activation functions, etc.
    * `train.json`: it's training related configuration, such as learning rate, batch size, etc.
    * `gaudi_config.json`(optional): it's only needed if you want to launch training on Gaudi, it contains some Gaudi releated params, see definitions [here](https://huggingface.co/docs/optimum/habana/package_reference/gaudi_config)
* **High Scability, plugable datasets and models**: we support highly customized functionality, such as:
    * Easy to transfer to users' own models: currently we only support Pytorch based models.
    * Easy to transfer to users' own datasets (in progress)
* **Native support for Intel Hardware**: we support intel hardware by using intel software stack, such as:
    * Intel® Habana 2nd gen Gaudi AI Accelerators 
    * Intel® Xeon® Scalable processors
    * Intel® Data Center GPU


## Validated Hardware Details

|Supported Hardware	| distributed support |
|--| -- | 
|Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| yes |
|Intel® Habana 2nd gen Gaudi AI Accelerators| no |


## Getting Started

#### 1. Prerequisites
For Intel GPU, ensure the [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) is installed.

For Gaudi, ensure the [SynapseAI SW stack and container runtime](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html?highlight=installer#run-using-containers) is installed.

#### 2. Clone the repository
```bash
git clone https://github.com/intel-sandbox/recsys-refkits.git rec-kit
cd rec-kit
```

#### 3. prepare the run time enviroment
##### (Option 1) run on bare mental environment 
* Use pip to install dependencies
```bash
# install base dependencies
pip install -e .
# install pytorch-releated dependencies
pip install -e .[pytorch]
# install gnn-releated dependencies
pip install -e .[gnn]
# install classical machine learning releated dependencies
pip install -e .[classical_ml]
```

##### (Option 2) run on docker environment
* Take Gaudi as an example, first, build the docker image
```bash
sudo docker build ./  \
    -f docker/Dockerfile.habana -t gaudi2_recsys:v1 \
    --build-arg dependencies_tag=pytorch \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} 
```
* After the image is built successfully, start a container:

```bash
# Add --cap-add sys_ptrace to enable py-spy in container if you need to debug.
# Set HABANA_VISIBLE_DEVICES if multi-tenancy is needed, such as "-e HABANA_VISIBLE_DEVICES=0,1,2,3"
sudo docker run \
    --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice --net=host \
    --name="gaudi_train" \
    --privileged \
    -v $(pwd):/workspace \
    -w /workspace/  \
    --device=/dev/dri \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=$no_proxy \
    -itd gaudi2_recsys:v1
```

#### 4. Running the usecase
* First, navigated to the specific usecase folder, take usecase 4 as example here
```bash
cd usecases/4_transformer_on_app_installation_pred
```
* Then, download the data and perform preprocessing
```bash
bash scripts/download_data.sh && bash scripts/data_prepare.sh
```
* Finally, lauch the training, we provide two options:
    * option 1: pure config based training
    ```bash 
    python -m recsys_kit.train config/train.json
    ```
    * option 2: command line based training, which use args passed through cmd line
    ```bash
    python -m recsys_kit.train \
    --model_name_or_path config/config.json \
    --dataset_local_path processed_data/sharechat_local \
    --metrics roc_auc \
    --output_dir checkpoint/tmp \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --warmup_steps 1000 \
    --per_device_train_batch_size 2048 \
    --per_device_eval_batch_size 4096 \
    --do_train \
    --do_eval \
    --lr_scheduler_type cosine \
    --weight_decay 0 \
    --save_strategy steps \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --dataloader_num_workers 8 \
    --use_hpu $use_hpu \
    --gaudi_config_name_or_path config/gaudi_config.json 
    ```

#### 5. Serving the RecSys pipeline

## Advanced usage
### 1. Replace with your models

### 2. Replace with your dataset


## How It works
<p align="center">
<img src=doc/image/TrainingWorkflow.png alt="train workflow" width="70%" height="50%">
</p>

## Reference Use Cases

| Use Case                                             | Task Type                            | Models | Dataset |
|------------------------------------------------------|--------------------------------------|--------|---------|
| [1_GBDT_on_app_installation_pred](usecases/1_GBDT_on_app_installation_pred) | CTR Prediction    | GBDT   | [sharechat](https://sharechat.com/recsys2023) |
| [2_shopping_cart_purchase_item_prediction](usecases/2_shopping_cart_purchase_item_pred) | Next Item Prediction    | GNN      | [Dressipi 1M Fashion Sessions Dataset](https://dressipi.com/downloads/recsys-datasets/) |
| [3_next_click_article_prediction](usecases/3_next_click_article_pred) | News Recommendation|PLM-NR| [Ekstra Bladet News Recommendation Dataset (EB-NeRD)](https://recsys.eb.dk/) |
| [4_transformer_on_app_installation_pred](usecases/4_transformer_on_app_installation_pred) | CTR Prediction          | Transformer | [sharechat](https://sharechat.com/recsys2023) |

### Highlights of Recsys Reference Use Case

* [ACM Digtal - SIHG4SR: Side Information Heterogeneous Graph for Session Recommender](https://dl.acm.org/doi/abs/10.1145/3556702.3556852)
* [ACM Digital - Graph Enhanced Feature Engineering for Privacy Preserving Recommendation Systems](https://dl.acm.org/doi/pdf/10.1145/3626221.3627290)
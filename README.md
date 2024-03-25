# Finetuning Large Language Models Efficiently on a Distributed Cluster

This repository is a boilerplate/fingerprint to fine tune any HuggingFace Large Language Model, such as Llama-2-7b, using a distributed cluster.
The purpose of this repo is to make it straightforward to fine tune any model efficiently by leveraging multi-GPU training.
It uses Ray AIR to orchestrate the cluster on AWS, and DeepSpeed for parameter+optimizer sharding + offloading.

The following Llama-2-7b model was fine-tuned using this repo: [https://huggingface.co/meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)

## Setup

First, you need to clone the repo:
```bash
git clone https://github.com/younesselbrag/GPUs-DeepSpeed-Distrub-LLM-
```
Then, configure your aws credentials using the `awscli` package command `aws configure`. This will allow Ray to spawn the head node and provision workers with the auto-scaling mechanism. If you don't have `awscli`, you can install it using `pip install awscli`.

## Working with the Ray cluster and submitting finetuning jobs

To spawn the cluster, simply run:

`ray up ray_cluster.yaml`

Once Ray has finished setting up the cluster, you can attach to the head node by doing:

`ray attach ray_cluster.yaml`

Now, to run a finetuning job, you can use the script `finetune.py` under `/src`.

An example usage is as below:

```bash
`python finetune.py --model="meta-llama/Llama-2-7b" --num-workers 4 --data alpaca_data_cleaned.json`
```

This will run a finetuning on the Llama-2-7b model using 4 GPU workers, and the Alpaca instruction dataset. Feel free to adjust the arguments for your own purposes.

When you are finished, you can turn off the cluster with:

`ray down ray_cluster.yaml`

## Changing DeepSpeed configuration

To tune the DeepSpeed configuration for your specific use case, edit the file on `config/deepspeed.json`. If you want to disable DeepSpeed, you can pass the `--no-deepspeed` parameter to the `finetune.py` script.

# Datasets

fine-tuned Llama-2-7b on the following 2 datasets:

- Alpaca: [https://huggingface.co/datasets/yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- Alpaca Spanish: [https://huggingface.co/datasets/bertin-project/alpaca-spanish](https://huggingface.co/datasets/bertin-project/alpaca-spanish)


## Credit 

[Large Language Models Efficiently on a Distributed Cluster](https://github.com/AdrianBZG/LLM-distributed-finetune/tree/main)

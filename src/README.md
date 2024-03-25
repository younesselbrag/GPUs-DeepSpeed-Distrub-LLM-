## Running the Script Funetune and generate

This script is designed to fine-tune a language model using Ray for distributed training. Follow the steps below to run the script:

### Prerequisites

- Python 3.6 or higher installed on your system.
- [Ray](https://docs.ray.io/en/latest/installation.html) installed (`pip install ray`).
- [Hugging Face Transformers](https://huggingface.co/transformers/installation.html) installed (`pip install transformers`).
- [Accelerate](https://huggingface.co/docs/accelerate/installation.html) installed (`pip install accelerate`).
- [PEFT](https://github.com/huggingface/peft) installed (`pip install peft`).
- [MLflow](https://www.mlflow.org/docs/latest/installation.html) installed (`pip install mlflow`).
- Other required dependencies listed in the `requirements.txt` file.

### Running the Script

1. Clone the repository to your local machine:

```bash
git clone https://github.com/younesselbrag/GPUs-DeepSpeed-Distrub-LLM-

cd GPUs-DeepSpeed-Distrub-LLM-
```

### Install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure that your data file is located in the data directory, or provide the path to your data file using the --data argument when running the script.

Run the script with the desired command-line arguments:

```bash
python funetune.py --num-workers 4 --use-cpu --no-deepspeed --model meta-llama/Llama-2-7b
```
Replace finetune.py with the name of your Python script, <number_of_workers> with the desired number of workers for training, and use the --use-cpu flag to enable CPU training if needed. You can also specify other optional arguments:

```YAML
    --num-workers: Sets the number of workers for training (default is 2).
    --use-cpu: Enables CPU training.
    --no-deepspeed: Disables DeepSpeed strategy.
    --model: Specifies the model from Hugging Face to use (default is "tiiuae/falcon-7b").
```
Monitor the training progress and metrics logged to MLflow during the training process.

After training is complete, the trained model will be saved in the trained_model directory.

Shutdown the Ray cluster after training by pressing Ctrl + C or running:

```bash
ray stop
```


## Load Save mode Generate LLM Context


The `generate.py` script is used to generate responses from a pre-trained language model using the Hugging Face Transformers library. Follow the steps below to run the script:

## Prerequisites

- Python 3.6 or higher installed on your system.
- [PyTorch](https://pytorch.org/get-started/locally/) installed.
- [Transformers](https://huggingface.co/transformers/installation.html) library installed (`pip install transformers`).
- A pre-trained language model available on the Hugging Face model hub.

## Running the Script

1. Clone the repository containing the script to your local machine:

```bash
git clone https://github.com/your_repository.git
cd your_repository
```

##### Run the script with the desired command-line arguments:

```bash

python generate.py --model <model_name_or_path> --instruction "Instruction text" [--input "Input text"] [--use-gpu] [--temperature <value>] [--top-p <value>] [--num-beams <value>]
```

Replace <model_name_or_path> with the name or path of the pre-trained language model available on the Hugging Face model hub. Provide the instruction text using the --instruction argument. Optionally, you can specify input text using the --input argument.

```YAML
Optional arguments:

    --use-gpu: Use this flag to enable GPU for inference. By default, the script uses CPU.
    --temperature <value>: Specifies the temperature for generation (default is 0.2).
    --top-p <value>: Specifies the top-p for generation (default is 0.75).
    --num-beams <value>: Specifies the num_beams for generation (default is 4).
```
The script will generate responses based on the provided instruction and input (if any) using the specified model. The generated response will be printed to the console.

You can experiment with different models, instructions, and input texts by modifying the command-line arguments accordingly.
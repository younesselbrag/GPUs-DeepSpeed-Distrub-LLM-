### DeepSpeed Intro

is a PyTorch optimization library that makes distributed training memory-efficient and fast. At it’s core is the Zero Redundancy Optimizer (ZeRO) which enables training large models at scale. ZeRO works in several stages:

1. ZeRO-1, optimizer state partioning across GPUs
2. ZeRO-2, gradient partitioning across GPUs
3. ZeRO-3, parameteter partitioning across GPUs

In GPU-limited environments, ZeRO also enables offloading optimizer memory and computation from the GPU to the CPU to fit and train really large models on a single GPU. DeepSpeed is integrated with the Transformers Trainer class for all ZeRO stages and offloading. All you need to do is provide a config file or you can use a provided template. For inference, Transformers support ZeRO-3 and offloading since it allows loading huge models.

This guide will walk you through how to deploy DeepSpeed training, the features you can enable, how to setup the config files for different ZeRO stages, offloading, inference, and using DeepSpeed without the Trainer.

##### Hyper-Paramters 

# Configuration Documentation

documentation for the configuration options in the provided JSON file.

## fp16
- **enabled**: Determines whether mixed precision training (FP16) is enabled. Can be set to "auto" to let the system decide automatically.
- **loss_scale**: Initial loss scaling factor for FP16 training.
- **loss_scale_window**: Window size for dynamic loss scaling.
- **initial_scale_power**: Initial scaling power for dynamic loss scaling.
- **hysteresis**: Hysteresis value for dynamic loss scaling.
- **min_loss_scale**: Minimum loss scaling factor.

## optimizer
- **type**: Type of optimizer used. For example, "AdamW".
- **params**:
  - **lr**: Learning rate for the optimizer. Can be set to "auto".
  - **betas**: Tuple of coefficients used for computing running averages of gradient and its square. Can be set to "auto".
  - **eps**: Term added to the denominator to improve numerical stability. Can be set to "auto".
  - **weight_decay**: Weight decay (L2 penalty) coefficient. Can be set to "auto".

## scheduler
- **type**: Type of learning rate scheduler. For example, "WarmupLR".
- **params**:
  - **warmup_min_lr**: Minimum learning rate during warmup.
  - **warmup_max_lr**: Maximum learning rate during warmup.
  - **warmup_num_steps**: Number of warmup steps.
  - **warmup_type**: Type of warmup schedule, e.g., linear.

## zero_optimization
- **stage**: Stage of ZeRO optimization.
- **offload_optimizer**: Configuration for offloading optimizer state.
- **offload_param**: Configuration for offloading model parameters.
- **overlap_comm**: Determines whether communication overlap is enabled.
- **contiguous_gradients**: Ensures contiguous gradients during ZeRO optimization.
- **reduce_bucket_size**: Size of reduction bucket. Can be set to "auto".
- **stage3_prefetch_bucket_size**: Size of prefetch bucket for stage 3.
- **stage3_param_persistence_threshold**: Threshold for parameter persistence in stage 3.
- **sub_group_size**: Sub-group size for communication.
- **stage3_max_live_parameters**: Maximum number of live parameters in stage 3.
- **stage3_max_reuse_distance**: Maximum reuse distance in stage 3.
- **stage3_gather_16bit_weights_on_model_save**: Determines whether 16-bit weights are gathered on model save.

## Other Configurations
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating weights.
- **gradient_clipping**: Threshold for gradient clipping. Can be set to "auto".
- **steps_per_print**: Frequency of printing training steps.
- **train_batch_size**: Batch size for training. Can be set to "auto".
- **train_micro_batch_size_per_gpu**: Micro batch size per GPU for training. Can be set to "auto".
- **wall_clock_breakdown**: Determines whether to breakdown the wall clock time for training.

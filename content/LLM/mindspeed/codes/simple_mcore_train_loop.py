import os
import torch
import mindspeed.megatron_adaptor 
from torch.optim import Adam
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

from megatron.core.transformer.module import Float16Module
from megatron.core import parallel_state
from megatron.core import dist_checkpointing
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.datasets.utils import compile_helpers 
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training import get_model
from megatron.training.tokenizer.tokenizer import _NullTokenizer

"""
torchrun --nproc-per-node 2  run_simple_mcore_train_loop.py
"""

# your_worker.py 或子进程函数中
import debugpy

def setup_debugpy():
    """在每个 rank 中设置 debugpy"""
    rank = int(os.environ.get("RANK", 0))
    # 使用 5678 + rank 作为端口（确保不冲突）
    debugpy_port = 22333 + rank
    
    print(f"[Rank {rank}] Waiting for debugger attach on port {debugpy_port}...")
    
    debugpy.listen(("0.0.0.0", debugpy_port))  # 绑定所有接口（兼容容器/远程）
    debugpy.wait_for_client()  # 阻塞直到 VS Code 连接
    
    print(f"[Rank {rank}] Debugger attached!")

_SEQUENCE_LENGTH = 64

def get_model_dtype(model):
    """
    Get the dtype of the model parameters
    """
    # Get the dtype of the first parameter
    for param in model.parameters():
        return param.dtype
    return None

def print_model_parameters(model, filename="model_parameters_info.txt"):
    """
    Print detailed information about model dtypes to a file
    Each line contains: parameter_name, shape, dtype, device
    """
    # Only run on rank 0 to avoid duplicate outputs
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    
    print(f"Writing model parameter information to {filename}...")
    
    with open(filename, 'w') as f:
        dtypes = set()
        devices = set()
        total_params = 0
        
        for name, param in model.named_parameters():
            dtypes.add(param.dtype)
            devices.add(param.device)
            total_params += param.numel()
            
            # Write parameter info to file: name, shape, dtype, device
            f.write(f"{name}, {list(param.shape)}, {param.dtype}, {param.device}\n")
        
        # Write summary at the end of file
        f.write(f"\nSummary:\n")
        f.write(f"Unique dtypes in model: {dtypes}\n")
        f.write(f"Unique devices in model: {devices}\n")
        f.write(f"Total parameters: {total_params}\n")
        
    print(f"Model parameter information written to {filename}")

def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    
    # worker_function(rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)

def model_provider():
    """
    Build the model.

    Note:
        1. 正常来说，这里配置参数fp16是控制训练采用fp16精度，但是这个函数中gpt_model没有被megetron.training.get_model
            函数所调用，所以这里配置参数fp16对训练没有影响，采用的默认fp32精度
        2. 即使这里设置了params_dtypes=fp16或者bf16，但是实际上gpt_model的部分参数精度还是fp32，比如position_embeddings.weight，
            layernorm的weight和bias
    """
    transformer_config = TransformerConfig(
        num_layers=2, 
        hidden_size=12, 
        num_attention_heads=4, 
        use_cpu_initialization=True,
        # pipeline_dtype=torch.float32,
        params_dtype=torch.float32, # 控制模型参数类型
        fp16=True, # 决定训练的前向反向的运算数据类型，没有被megetron.training.get_model函数调用，无效果
    )

    print("Creating GPT model...")
    gpt_model = GPTModel(
        config=transformer_config, 
        transformer_layer_spec=get_gpt_layer_local_spec(), 
        vocab_size=100, 
        max_sequence_length=_SEQUENCE_LENGTH,
    )    
    print(gpt_model)
    # setup_debugpy()
    print("GPT model created.")
    
    def hook_fn(module, input, output):
        print("Layer input dtype: ", input[0].dtype)
        print("weight dtype in layer: ", next(module.parameters()).dtype)
    
    list(gpt_model.modules())[2].register_forward_hook(hook_fn)
    print_model_parameters(gpt_model)
    return gpt_model

def get_train_data_iterator():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=_SEQUENCE_LENGTH),
    )

    print("Building datasets...")
    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()
    print("Datasets built.")

    print("Creating data loader...")
    train_dataloader = DataLoader(datasets[0], batch_size=8, shuffle=True)
    print("Data loader created.")

    train_iterator = iter(train_dataloader)
    print("Train iterator ready.")
    
    return train_iterator
def forward_step_func(data_iterator, model):
    """
    自定义前向函数

    Notes:
        1. model(tokens, position_ids, attention_mask, labels=labels)返回的是loss，而不是logits
        2. 要返回logits，则仅需要拿掉labels参数即可
    """
    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    data = next(data_iterator)
    tokens = data['tokens'].to(device)
    attention_mask = data['attention_mask'].to(device)
    position_ids = data['position_ids'].to(device)
    labels = data['labels'].to(device)
    loss_mask = data['loss_mask'].to(device)

    # output loss
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)
    
    # # output logits
    # output_tensor = model(tokens, position_ids, attention_mask)

    # Store the logits (output_tensor before loss computation)
    global forward_logits
    forward_logits = output_tensor.detach()  # Detach to avoid gradient tracking
    return output_tensor, partial(loss_func, loss_mask)

def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)

def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict=gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
    gpt_model.load_state_dict(checkpoint)
    return gpt_model

if __name__ == "__main__":
    initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    
    model_parallel_cuda_manual_seed(123)

    print("Creating model...")
    gpt_model = model_provider()
    print("Model created successfully.")
    
    device = torch.device("cuda")
    gpt_model.to(device)
    print("Model moved to GPU.")

    optim = Adam(gpt_model.parameters())

    print("Preparing training data iterator...")
    train_iterator = get_train_data_iterator()
    print("Training data iterator ready.")

    print("Getting forward-backward function...")
    forward_backward_func = get_forward_backward_func()
    print("Forward-backward function acquired.")

    forward_logits = None
    # Running the model for 5 iterations
    for i in range(5):
        print(f"Starting iteration {i+1}/5...")
        optim.zero_grad()
        print("Gradients zeroed.")
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=8,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False)
        print(f"Forward-backward pass completed with losses: {losses_reduced}")
        if forward_logits is not None:
            print(f"Logits shape: {forward_logits.shape}")
        optim.step()
        print("Optimizer step completed.")

        print(f'Losses reduced :  {losses_reduced}')
        print(f"Iteration {i+1}/5 completed.\n")
    # Saving the model
    print("Saving model checkpoint...")
    ckpt_path = os.getcwd() + '/ckpt'
    Path(ckpt_path).mkdir(exist_ok=True)
    save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)
    print("Model checkpoint saved.")

    # Loading the model
    print("Loading model checkpoint...")
    gpt_model = load_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)
    gpt_model.to(device)
    print('Successfully loaded the model')


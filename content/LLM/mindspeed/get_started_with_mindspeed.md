---
title: "开始使用 Mindspeed"
date: 2025-11-11
---

本指南通过[示例代码](/llm/mindspeed/codes/simple_mcore_train_loop.py)演示如何使用 Megatron 进行 Mindspeed 训练。

## Mindspeed 并行环境初始化

作为一个分布式的大模型训练框架，`Mindspeed` 在初始化期间需要设置分布式环境。核心初始化代码如下：

```python
import torch
import mindspeed.megatron_adaptor  # 导入适配器以确保 Mindspeed 与 Megatron 兼容
from megatron.core import parallel_state

def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    parallel_state.destroy_model_parallel()

    # 标准 PyTorch 分布式训练设置
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron 特定的分布式训练初始化
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)
```

与典型的并行代码的关键区别在于，除了调用标准的 torch.distributed.init_process_group 外，Mindspeed 还需要 parallel_state 初始化模型并行。注意在parallel_state初始化中，需要传入TP和PP的大小。

此外，还需要注意，添加代码行`import mindspeed.megatron_adaptor`来保证Mindspeed对Megatron的API进行兼容。

## Mindspeed 模型初始化

```python

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel

def model_provider():
    """
    Build the model.
    """
    transformer_config = TransformerConfig(
        num_layers=2, 
        hidden_size=12, 
        num_attention_heads=4, 
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float16, # 控制模型参数类型
        bf16=True, # 决定训练的前向反向的运算数据类型，没有被megetron.training.get_model函数调用，无效果
    )

    print("Creating GPT model...")
    gpt_model = GPTModel(
        config=transformer_config, 
        transformer_layer_spec=get_gpt_layer_local_spec(), 
        vocab_size=100, 
        max_sequence_length=_SEQUENCE_LENGTH,
    )    
    print(gpt_model)
    print("GPT model created.")
    return gpt_model
```

可以看到模型初始化分为两步：
1. Transformer的配置参数，包括transformer layer层数，hidden size，attention heads数，模型参数数据类型。
2. 模型结构，包括transformer layer层，以及vocab size和max sequence length。

上面这两步基本可以自定义一个Transformer网络的结构了。可以查看模型打印出来的结构：

```bash
GPTModel(
  (embedding): LanguageModelEmbedding(
    (word_embeddings): VocabParallelEmbedding()
    (position_embeddings): Embedding(64, 12)
    (embedding_dropout): Dropout(p=0.1, inplace=False)
  )
  (decoder): TransformerBlock(
    (layers): ModuleList(
      (0-1): 2 x TransformerLayer(
        (input_layernorm): FusedLayerNorm()
        (self_attention): SelfAttention(
          (core_attention): DotProductAttention(
            (scale_mask_softmax): FusedScaleMaskSoftmax()
            (attention_dropout): Dropout(p=0.1, inplace=False)
          )
          (linear_proj): RowParallelLinear(in_features=12, out_features=12, bias=False, TP=1)
          (linear_qkv): ColumnParallelLinear(in_features=12, out_features=36, bias=False, TP=1)
          (q_layernorm): IdentityOp()
          (k_layernorm): IdentityOp()
        )
        (pre_cross_attn_layernorm): IdentityOp()
        (cross_attention): IdentityOp()
        (cross_attn_bda): IdentityFuncOp()
        (pre_mlp_layernorm): FusedLayerNorm()
        (mlp): MLP(
          (linear_fc1): ColumnParallelLinear(in_features=12, out_features=48, bias=False, TP=1)
          (linear_fc2): RowParallelLinear(in_features=48, out_features=12, bias=False, TP=1)
        )
      )
    )
    (final_layernorm): FusedLayerNorm()
  )
  (output_layer): ColumnParallelLinear(in_features=12, out_features=100, bias=False, TP=1)
)
```

同时，我们也可以打印出模型参数的信息(TP=PP=1)：

```bash
embedding.word_embeddings.weight, [100, 12], torch.float16, cpu
embedding.position_embeddings.weight, [64, 12], torch.float32, cpu
decoder.layers.0.input_layernorm.weight, [12], torch.float32, cpu
decoder.layers.0.input_layernorm.bias, [12], torch.float32, cpu
decoder.layers.0.self_attention.linear_proj.weight, [12, 12], torch.float16, cpu
decoder.layers.0.self_attention.linear_proj.bias, [12], torch.float16, cpu
decoder.layers.0.self_attention.linear_qkv.weight, [36, 12], torch.float16, cpu
decoder.layers.0.self_attention.linear_qkv.bias, [36], torch.float16, cpu
decoder.layers.0.pre_mlp_layernorm.weight, [12], torch.float32, cpu
decoder.layers.0.pre_mlp_layernorm.bias, [12], torch.float32, cpu
decoder.layers.0.mlp.linear_fc1.weight, [48, 12], torch.float16, cpu
decoder.layers.0.mlp.linear_fc1.bias, [48], torch.float16, cpu
decoder.layers.0.mlp.linear_fc2.weight, [12, 48], torch.float16, cpu
decoder.layers.0.mlp.linear_fc2.bias, [12], torch.float16, cpu
decoder.layers.1.input_layernorm.weight, [12], torch.float32, cpu
decoder.layers.1.input_layernorm.bias, [12], torch.float32, cpu
decoder.layers.1.self_attention.linear_proj.weight, [12, 12], torch.float16, cpu
decoder.layers.1.self_attention.linear_proj.bias, [12], torch.float16, cpu
decoder.layers.1.self_attention.linear_qkv.weight, [36, 12], torch.float16, cpu
decoder.layers.1.self_attention.linear_qkv.bias, [36], torch.float16, cpu
decoder.layers.1.pre_mlp_layernorm.weight, [12], torch.float32, cpu
decoder.layers.1.pre_mlp_layernorm.bias, [12], torch.float32, cpu
decoder.layers.1.mlp.linear_fc1.weight, [48, 12], torch.float16, cpu
decoder.layers.1.mlp.linear_fc1.bias, [48], torch.float16, cpu
decoder.layers.1.mlp.linear_fc2.weight, [12, 48], torch.float16, cpu
decoder.layers.1.mlp.linear_fc2.bias, [12], torch.float16, cpu
decoder.final_layernorm.weight, [12], torch.float32, cpu
decoder.final_layernorm.bias, [12], torch.float32, cpu
output_layer.weight, [100, 12], torch.float16, cpu

Summary:
Unique dtypes in model: {torch.float16, torch.float32}
Unique devices in model: {device(type='cpu')}
Total parameters: 6960
```

注意观察我们设置的参数与上面参数维度的对应关系：
1. vocab_size: 100
2. hidden_size: 12
3. sequence_length: 64
4. num_attention_heads: 4

还可以发现，虽然参数类型设置的是float16，但是模型参数并不都是float16，有部分网络参数仍然为float32，特别是layernorm的网络层。

我们也可以打印出其它并行配置下模型参数的信息(TP=2, PP=1)：

```bash
embedding.word_embeddings.weight, [50, 12], torch.float16, cpu
embedding.position_embeddings.weight, [64, 12], torch.float32, cpu
decoder.layers.0.input_layernorm.weight, [12], torch.float32, cpu
decoder.layers.0.input_layernorm.bias, [12], torch.float32, cpu
decoder.layers.0.self_attention.linear_proj.weight, [12, 6], torch.float16, cpu
decoder.layers.0.self_attention.linear_proj.bias, [12], torch.float16, cpu
decoder.layers.0.self_attention.linear_qkv.weight, [18, 12], torch.float16, cpu
decoder.layers.0.self_attention.linear_qkv.bias, [18], torch.float16, cpu
decoder.layers.0.pre_mlp_layernorm.weight, [12], torch.float32, cpu
decoder.layers.0.pre_mlp_layernorm.bias, [12], torch.float32, cpu
decoder.layers.0.mlp.linear_fc1.weight, [24, 12], torch.float16, cpu
decoder.layers.0.mlp.linear_fc1.bias, [24], torch.float16, cpu
decoder.layers.0.mlp.linear_fc2.weight, [12, 24], torch.float16, cpu
decoder.layers.0.mlp.linear_fc2.bias, [12], torch.float16, cpu
decoder.layers.1.input_layernorm.weight, [12], torch.float32, cpu
decoder.layers.1.input_layernorm.bias, [12], torch.float32, cpu
decoder.layers.1.self_attention.linear_proj.weight, [12, 6], torch.float16, cpu
decoder.layers.1.self_attention.linear_proj.bias, [12], torch.float16, cpu
decoder.layers.1.self_attention.linear_qkv.weight, [18, 12], torch.float16, cpu
decoder.layers.1.self_attention.linear_qkv.bias, [18], torch.float16, cpu
decoder.layers.1.pre_mlp_layernorm.weight, [12], torch.float32, cpu
decoder.layers.1.pre_mlp_layernorm.bias, [12], torch.float32, cpu
decoder.layers.1.mlp.linear_fc1.weight, [24, 12], torch.float16, cpu
decoder.layers.1.mlp.linear_fc1.bias, [24], torch.float16, cpu
decoder.layers.1.mlp.linear_fc2.weight, [12, 24], torch.float16, cpu
decoder.layers.1.mlp.linear_fc2.bias, [12], torch.float16, cpu
decoder.final_layernorm.weight, [12], torch.float32, cpu
decoder.final_layernorm.bias, [12], torch.float32, cpu
output_layer.weight, [50, 12], torch.float16, cpu

Summary:
Unique dtypes in model: {torch.float32, torch.float16}
Unique devices in model: {device(type='cpu')}
Total parameters: 3948
```

可以看到，有的参数的维度减半了！

另外，我们注意到在这里，我们设置`bf16=True`，去检查模型的参数，或者debug模型中间层的输入输出，其类型依然是float32。这是因为我们这里没有调用megatron.training.get_model函数，而是直接调用了megatron.model.GPTModel，这里GPTModel没有对模型的数值类型做任何的处理，仅仅是通过para_dtype初始化了模型参数。而get_model函数里面，则会根据bf16参数，对模型参数进行类型转换：

```python
from megatron.core.enums import ModelType

def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    ...
    # Fp16 conversion.
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]
    ...
```

这里，我们还可以查看Float16Module的实现，该模块将模型转换为半精度浮点数。

```python
class Float16Module(MegatronModule):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(Float16Module, self).__init__(config)
        self.config = config
        self.fp16 = config.fp16
        self.bf16 = config.bf16

        if self.fp16:
            self.add_module('module', module.half())

            def float16_convertor(val):
                return val.half()

        elif self.bf16:
            self.add_module('module', module.bfloat16())

            def float16_convertor(val):
                return val.bfloat16()

        else:
            raise Exception('Either config.fp16 or config.bf16 should be True.')

        self.float16_convertor = float16_convertor

    def set_input_tensor(self, input_tensor):
        return self.module.set_input_tensor(input_tensor)

    def forward(self, *inputs, **kwargs):
        if parallel_state.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if parallel_state.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Retrieve state_dict from the module being wrapped."""
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(self, prefix='', *args, **kwargs):
        """Retrieve sharded_state_dict from the module being wrapped."""
        return self.module.sharded_state_dict(prefix, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)

```

## MindSpeed 训练主循环入口

在基于Megatron的模型训练中，我们经常看到这样的训练循环：

```python
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

optim = Adam(gpt_model.parameters())
forward_backward_func = get_forward_backward_func()
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
    optim.step()
```

可以看到，`forward_backward_func`是`megatron`内置的包装好的函数，其内置了前向和反向的计算过程，我们需要做的就是传入自定义的`forward_step_func`和`data_iterator`和`model`。输出则是reduced的loss，其shape为[bs, seq_len]，即每一个token的loss。

自定义的前向函数，主要包括了对输入数据的简单预处理和loss函数的定义：

```python
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
    
    return output_tensor, partial(loss_func, loss_mask)
```

注意，前向函数返回的是模型输出的每一个token的loss，和loss函数的偏函数。关于loss的计算，大模型给出了一个解答：

```markdown
整个loss计算分为两个阶段：

**第一阶段（在模型内部）**：

- 模型自己计算基础的token级别loss
- 返回形状为 [batch_size, sequence_length] 的tensor

**第二阶段（在自定义loss_func中）**：

- 对模型返回的loss进行进一步处理
- 应用loss_mask进行过滤
- 计算最终的平均loss
```

## 调试Mindspeed训练

Mindspeed训练代码通常是多机多卡的，因此调试起来可能会比较麻烦。这里提供一个成熟验证的方法来debug。

1. 在训练代码中，添加setup_debugpy()函数

```python
import os
def setup_debugpy():
    """在每个 rank 中设置 debugpy"""
    rank = int(os.environ.get("RANK", 0))
    # 使用 5678 + rank 作为端口（确保不冲突）
    debugpy_port = 22333 + rank
    
    print(f"[Rank {rank}] Waiting for debugger attach on port {debugpy_port}...")
    
    debugpy.listen(("0.0.0.0", debugpy_port))  # 绑定所有接口（兼容容器/远程）
    debugpy.wait_for_client()  # 阻塞直到 VS Code 连接
    
    print(f"[Rank {rank}] Debugger attached!")
```

2. 配置debug文件,.vscode/launch.json里面写入配置：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to Worker 0",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 22333
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "."  // 如果本地/远程路径一致
        }
      ]
    },
    {
      "name": "Attach to Worker 1",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 22334
      }
    }
  ]
  ...
}

```

3. 正常启动Mindspeed的训练

```bash
torchrun --nproc-per-node 4 mindspeed_train.py 
```

4. vscode调试器链接代码

当终端打印出```[Rank 0] Waiting for debugger attach on port 22333...```，则可以点击vscode的debugger图标，选择Attach to Worker 1，然后点击启动按钮。

---
title: "开始使用 Mindspeed 进行大模型训练"
date: 2025-11-11
---

本文通过一个[完整示例代码](https://github.com/fandengdong/fdd.github.io/blob/main/content/LLM/mindspeed/codes/simple_mcore_train_loop.py)演示如何使用 **Mindspeed** 框架进行分布式大模型训练。

> **环境版本信息**  
> - Mindspeed commit ID: `89f4632d`  
> - Megatron 分支: `core_v0.12.1`  
> - CANN: `8.2.RC1`  
> - PyTorch: `2.5.1`

---

## 1. 初始化分布式并行环境

Mindspeed 基于 Megatron 构建，支持张量并行（TP）和流水线并行（PP）。在训练前，必须正确初始化分布式环境：

```python
import os
import torch
import mindspeed.megatron_adaptor  # 关键：确保 Mindspeed 与 Megatron API 兼容
from megatron.core import parallel_state

def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    # 清理已有状态（防止重复初始化）
    parallel_state.destroy_model_parallel()

    # 标准 PyTorch 分布式设置
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # 初始化 Megatron 并行状态
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size
    )
```

> **关键点说明**：
> 必须显式导入 mindspeed.megatron_adaptor，以启用兼容层。
> 除了标准的 torch.distributed.init_process_group，还需调用 parallel_state.initialize_model_parallel 来激活 TP/PP 支持。
> 需根据实际训练配置传入 tensor_model_parallel_size 和 pipeline_model_parallel_size。

## Mindspeed 模型初始化

Mindspeed 使用 Megatron Core 的模块化设计构建模型。以下是一个最小 GPT 模型的构建示例：

```python
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

_SEQUENCE_LENGTH = 64

def model_provider():
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float16,  # 控制模型参数存储类型
        bf16=True,                   # 控制前向/反向计算的数据类型（需配合 get_model 使用）
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

**模型结构解析**

模型创建分为两步：
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

可以看到，有的参数的维度减半了！这是因为TP并行，对模型的部分参数按照TP数进行了切分。仔细看，切分的网络层主要是：
1. embedding.word_embeddings.weight
2. self_attention.linear_qkv.weight，self_attention.linear_qkv.bias
3. mlp.linear_fc1.weight, mlp.linear_fc1.bias
4. output_layer.weight

如果我们采用PP切分，则能看到rank0进程的model只包含了前半部分的网络层，rank1进程的model包含后半部分。

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

这里，我们还可以查看Float16Module的实现，该模块在计算前向的时候，在forward函数中临时将模型转换为半精度浮点数，算完后，又转换为fp32精度。

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

    def forward(self, *inputs, **kwargs):
        if parallel_state.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if parallel_state.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
        return outputs
    ...

```

在实践中，使用混合精度训练时，想避免Float16Module对自定义的网络层的参数数据类型进行转换（主要是想保持fp32精度），我们需要同时重写网络层的half(self)，bfloat16(self)方法和_apply(self, fn)方法：

```python
class CustomLayer(nn.Module):
    ...
    def _apply(self, fn):
        return self

    def half(self):
        return self 

    def bfloat16(self):
        return self 
    ...
```

这样可以避免网络对自定义网络层的参数进行类型转换。另外，在前向的时候，还尽量显式的将输入数据转换为torch.float32。


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

`forward_backward_func`是`megatron`内置的包装好的函数，其内置了前向和反向的计算过程，我们需要做的就是传入自定义的`forward_step_func`和`data_iterator`和`model`。输出则是reduced的loss，其shape为[bs, seq_len]，即每一个token的loss。

自定义的前向函数`forward_backward_func`，主要包括了对输入数据的简单预处理和loss函数的定义：

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

3. 在合适的地方出入setup_debugpy()，放在启动了多进程之后的代码位置，参考[这里](https://github.com/fandengdong/fandengdong.github.io/blob/main/content/LLM/mindspeed/codes/simple_mcore_train_loop.py#L129)

4. 正常启动Mindspeed的训练

```bash
torchrun --nproc-per-node 4 mindspeed_train.py 
```

5. vscode调试器链接代码

当终端打印出```[Rank 0] Waiting for debugger attach on port 22333...```，则可以点击vscode的debugger图标，选择Attach to Worker 0，然后点击启动按钮。


## 关于mindspeed的训练精度

mindspeed提供了两个flag，用于设置训练精度：
1. `--fp16`: 使用fp16训练精度。fp16表示的数值范围比较小，可能会出现上溢或者下溢，采用fp16进行训练，mindspeed会自动调用loss scaler来处理梯度向下溢出的问题（注意loss scaler是自动在optimizer中处理的）。
2. `--bf16`: 使用bf16训练精度。bf16采用了更多的指数位，因此表示的数值范围会大很多，因此不需要loss scaler，但是bf16的数值精度要比fp16小。
3. 如果`--fp16`和`--bf16`都是false，则默认会使用fp32训练精度。

### 浮点数格式比较

#### FP16 格式  
FP16 是一种 16 位浮点数格式，包括：  
- **1 位**用于符号（S）  
- **5 位**用于指数（E）  
- **10 位**用于尾数（M，也称为分数或 mantissa）

| 字段   | 比特数 | 描述 |
|--------|--------|------|
| 符号 (S) | 1      | 表示数字的正负 |
| 指数 (E) | 5      | 使用偏置（bias）为 15，表示范围约为 [-14, 15] |
| 尾数 (M) | 10     | 隐含前导 1，实际有效精度为 11 位 |

---

#### BF16 格式  
BF16 也是一种 16 位浮点数格式，但比特分配不同：  
- **1 位**用于符号（S）  
- **8 位**用于指数（E）  
- **7 位**用于尾数（M）

| 字段   | 比特数 | 描述 |
|--------|--------|------|
| 符号 (S) | 1      | 表示数字的正负 |
| 指数 (E) | 8      | 使用偏置（bias）为 127，表示范围约为 [-126, 127] |
| 尾数 (M) | 7      | 有效精度较低，无隐含位优化（与 FP32 对齐） |

---

### 数值表示范围和精度对比

由于 BF16 将更多比特分配给指数部分，其**数值表示范围远大于 FP16**，接近 FP32；但尾数位更少，因此**精度低于 FP16**。

| 格式 | 总位数 | 指数位数 | 尾数位数 | 数值范围（近似）         | 精度       |
|------|--------|----------|----------|--------------------------|------------|
| FP16 | 16     | 5        | 10       | \(6.1 \times 10^{-5}\) 到 \(6.5 \times 10^{4}\) | 较高       |
| BF16 | 16     | 8        | 7        | 接近 FP32（约 \(10^{-38}\) 到 \(10^{38}\)） | 中等（低于 FP16） |

> 💡 **总结**：  
> - **FP16**：精度高，但动态范围小，训练时需配合 **loss scaler** 防止下溢。  
> - **BF16**：动态范围大（类似 FP32），无需 loss scaler，但数值精度较低，适合对范围敏感、对精度容忍度较高的深度学习训练场景。

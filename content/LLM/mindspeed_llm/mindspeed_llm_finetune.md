---
title: "使用 Mindspeed-LLM进行训练"
date: 2025-11-14
---

本指南记录采用mindspeed-llm进行训练的步骤，以微调为例。

## 准备数据

准备数据集的jsonl文件，每个样本是一个字典，至少要包含关键字`prompt`和`response`，其中`prompt`为输入，`response`为输出。

注意：如果提供的jsonl文件里面没有包含`chat template`，那么在转换数据的时候要添加`prompt-type`参数来提供模板。下面提供一个带了chat templated的jsonl文件样例：

```jsonl
{"prompt":"<|im_start|>system\nYou are a helpful assistant. To answer the user\'s question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> <\/think> and <answer> <\/answer> tags, respectively, i.e., <think> reasoning process here <\/think> <answer> answer here <\/answer>.<|im_end|>\n<|im_start|>user\nLet $\\triangle ABC$ have circumcenter $O$ and incenter $I$ with $\\overline{IA}\\perp\\overline{OI}$, circumradius $13$, and inradius $6$. Find $AB\\cdot AC$.\nPlease reason step by step, and put your final answer within \\boxed{}.\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n","response":"<think>\nI have a geometry problem. We have a triangle ABC with circumcenter O and incenter I...."}
{"prompt":"<|im_start|>system\nYou are a helpful assistant. To answer the user\'s question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> <\/think> and <answer> <\/answer> tags, respectively, i.e., <think> reasoning process here <\/think> <answer> answer here <\/answer>.<|im_end|>\n<|im_start|>user\nLet \\(b\\ge 2\\) be an integer. Call a positive integer \\(n\\) \\(b\\text-\\textit{eautiful}\\) if it has exactly two digits when expressed in base \\(b\\)  and these two digits sum to \\(\\sqrt n\\). For example, \\(81\\) is \\(13\\text-\\textit{eautiful}\\) because \\(81  = \\underline{6} \\ \\underline{3}_{13} \\) and \\(6 + 3 =  \\sqrt{81}\\). Find the least integer \\(b\\ge 2\\) for which there are more than ten \\(b\\text-\\textit{eautiful}\\) integers.\nPlease reason step by step, and put your final answer within \\boxed{}.\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n","response":"<think>\nThe problem defines a \"b-beautiful\" number as a positive integer..."}
```

准备好jsonl文件后，利用mindspeed-llm自带的脚本preprocess_data.py来预处理数据：

```bash
#!/bin/bash

source /home/fdd/workspace/Ascend/CANN8.2.RC1/ascend-toolkit/set_env.sh

python ./preprocess_data.py \
    --input /path/to/dataset.jsonl \
    --tokenizer-name-or-path /home/fdd/workspace/models/Qwen/Qwen2.5-32B \
    --output-prefix /path/to/dataset \
    --workers 64 \
    --n-subs 1 \
    --log-interval 100 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type empty  \
    --seq-length 65536 \
    --cache-dir /home/fdd/workspace/tmp \
    --map-keys '{"prompt":"prompt", "query":"", "response":"response"}' # 默认值，可不传

    # --pack \
    # --neat-pack \

# --map-keys '{"prompt":"prompt","query":"input","response":"answer"}' # 默认值，可不传
# --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传
# 32768
```

上面脚本有几个参数需要说明下：
1. `--output-prefix`: 这个`prefix`中`dataset`是**输出文件的前缀**，后续训练填写路径跟这个保持一致，好多人只填写到目录，导致没有读取数据集；
2. `--n-subs`: 这个参数是处理数据集的时候，将数据集切分为几分同时处理；一般对于比较大的数据集文件，可以开启8，速度会显著提升；
3. `--prompt-type`: 这个参数是设置模型训练的`chat template`，如果数据集自带了`chat template`，则设置为`empty`，否则需要自己从`configs/finetune/templates.json`里面挑选一个模板，或者自定义一个模板，并保存在`configs/finetune/templates.json`里面，然后设置为对应的模板名称；
4. `--seq-length`：这个参数是设置模型训练的`sequence length`，这个参数需要根据数据集和模型进行设置，一般设置为数据集的`max_seq_len`，或者模型默认的`max_seq_len`；
5. `--cache_dir`：这个参数是设置模型缓存目录，这个参数可以不设置。建议设置到存储空间大的位置；
6. `--map-keys`: 这个参数是设置数据集的映射关系，这个参数可以不设置。如果数据集的`key`和脚本需要的`key`不一致，可以通过这个参数进行映射；
7. 上面配置是默认采用动态长度的数据集处理方式。为了提升训练效率，可以采用packing的方法预处理数据集，添加`--pack`和`--neat-pack`参数即可；

## 准备权重

mindspeed-llm读取的权重跟megatron保持一致，即采用mcore权重。一般我们下载的模型为huggingface格式，因此需要将huggingface格式的权重转换成mcore格式。

mindspeed-llm一共了各种模型架构的权重转换脚本，具体可以在这个目录下找到：examples/mcore/，我们以Qwen2.5-7B模型为例：

```bash
# 修改 ascend-toolkit 路径
source /home/fdd/workspace/Ascend/CANN8.2.RC1/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

model_path_hf=/home/fdd/workspace/models/Qwen/Qwen2.5-7B
TP=1
PP=1

# 设置需要的权重转换参数
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size $TP \
       --target-pipeline-parallel-size $PP \
       --add-qkv-bias \
       --load-dir $model_path_hf \
       --save-dir $model_path_hf/mcore_tp${TP}_pp${PP}/ \
       --tokenizer-model $model_path_hf/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16 # --num-layer-list 11,13,19,21 参数根据需要添加

```

我根据个人使用习惯，稍微修改了一下官方脚本，将TP和PP参数放在一起，新转换的权重路径跟原始路径保持一致，方便后续使用。

## 开启训练

准备好数据集和训练权重后，我们可以利用mindspeed-llm提供的训练脚本进行训练。mindspeed-llm针对各个模型（deepseek，qwen，llama等）的各种任务（pretrain，finetune，chat，generate)提供了相应的训练脚本，这里以qwen为例，脚本位于`examples/mcore/qwen25/tune_qwen25_7b_4k_full_ptd.sh`

```bash
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=4     # 每个节点使用的NPU数量，必须大于等于TP*PP
MASTER_ADDR=localhost # 训练主节点的IP地址
MASTER_PORT=6000  # 训练主节点的端口号
NNODES=1  # 训练节点数量
NODE_RANK=0 # 训练节点的node_id，如果是多个节点训练，则每个节点的node_id必须从0开始递增
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) # 自动计算训练总共使用的卡数

# please fill these path configurations
CKPT_LOAD_DIR="/home/fdd/workspace/models/Qwen/Qwen2.5-7B/mcore_tp2_pp1/"
CKPT_SAVE_DIR="/home/fdd/workspace/models/Qwen/Qwen2.5-7B/mcore_tp2_pp1/ckpt"
DATA_PATH="/home/fdd/workspace/datasets/merged_long_COT_SFT/merged_SFT_final"
TOKENIZER_PATH="/home/fdd/workspace/models/Qwen/Qwen2.5-7B"

TP=4   # 模型并行数
PP=1   # pipeline并行数
SEQ_LEN=4096 # 模型训练的最长输入序列长度
MBS=1   # mindspeed-llm中前向和反向的micro BS，计算完，并不马上更新梯度
GBS=8   # 模型训练的global BS，也就是拿到8个样本的梯度后，才更新一次
TRAIN_ITERS=5000 # 训练步长

...

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl \
    | tee logs/tune_mcore_qwen25_7b_full.log

```

可以看到finetune的启动文件为posttrain_gpt.py，启动方式为torchrun，其它所有的参数都可以在bash启动脚本中配置。

## mindspeed-llm训练主流程分析

先看一下posttrain_gpt.py的启动脚本，这只是一个启动入口

```python
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from mindspeed_llm import megatron_adaptor
from mindspeed_llm.tasks.posttrain.launcher import AutoTrainer


def launch():
    trainer = AutoTrainer()
    trainer.train()

if __name__ == '__main__':
    launch()
```

继续查看AutoTrainer的实现，发现是一个分配任务的类：

```python

def get_trainer(stage):
    """
    Factory function to select the appropriate trainer based on the 'stage' argument.

    :param stage: A string representing the stage of the training.
    :return: An instance of the appropriate trainer class.
    """
    if stage == "sft":
        return SFTTrainer()
    elif stage == "dpo":
        return DPOTrainer()
    elif stage == "orm":
        return ORMTrainer()
    elif stage == "prm":
        return PRMTrainer()
    elif stage == "simpo":
        return SimPOTrainer()
    elif stage == "trl_ppo":
        return TrlPPOTrainer()
    else:
        logger.info(f'Unknown Stage: {stage}')
        return None


class AutoTrainer:
    """
    AutoTrainer is an automatic trainer selector.
    It chooses the appropriate trainer (e.g., SFTTrainer, DPOTrainer, ORMTrainer...)
    based on the 'stage' argument.
    """

    def __init__(self):
        """
        Initializes the AutoTrainer.

        - Initializes the training system.
        - Retrieves the 'stage' argument.
        - Uses the 'stage' to select the correct trainer.
        """
        initialize_megatron()
        self.args = get_args()
        self.trainer = get_trainer(self.args.stage)

    def train(self):
        """
        Starts the training process by invoking the 'train()' method of the selected trainer.
        """
        self.trainer.train()
```

因此，对于真正的训练，可以查看`SFTTrainer`的实现：

```python

class SFTTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""
        # Items and their type.
        ...

    @staticmethod
    def loss_func(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function.

        Args:
            input_tensor (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses
        """
        args = get_args()
        loss_mask = input_tensor

        ...

    def forward_step(self, data_iterator, model):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator)
        timers('batch-generator').stop()

        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask,
                                  labels=labels)
        else:
            output_tensor = model(tokens, position_ids, attention_mask,
                                  labels=labels, loss_mask=loss_mask)

        return output_tensor, partial(self.loss_func, loss_mask)
```

SFTTrainer主要定义了：
1. 数据预处理，从dataloader给到的数据如何正确的喂给模型；
2. 损失函数，如何计算模型输出的loss；
3. 前向函数，如何将数据输入模型，并返回模型输出和损失函数。

我们再看SFTTrainer的父类BaseTrainer：

```python

class BaseTrainer(ABC):
    """
    BaseTrainer is an abstract base class that provides fundamental functions for training large language models.
    
    It defines the following core methods:
    - `__init__`: Initializes the basic attributes of the trainer.
    - `initialize`: Initializes the trainer, including setting up timers, data iterators, etc.
    - `model_provider`: Provides the model to be trained.
    - `get_batch`: Retrieves a batch of data from the data iterator.
    - `loss_func`: Computes the loss function.
    - `forward_step`: Performs a forward pass step, computing the loss.
    - `train`: The main training loop, controlling the entire training process.

    """

    def __init__(self, process_non_loss_data_func=None):
        self.args = get_args()
        self.timers = get_timers()
        self.process_non_loss_data_func = process_non_loss_data_func
        self.train_args = None
        self.model_type = None
        self.test_data_iterator_list = None
        self.train_valid_test_datasets_provider = train_valid_test_datasets_provider
        self.initialize()
        ...

    def model_provider(self, pre_process, post_process):
        """
        Builds the model.

        If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
            Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """

        ...

        model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
                mtp_block_spec=mtp_block_spec,
            )
        ...

    def train(self):
        args = get_args()

        ...

        iteration, num_floating_point_operations_so_far = train(*self.train_args)

        ...
```

这里我们只关心GPTModel的定义和train函数。注意这里的GPTModel定义不是`megatron/core/models/gpt/gpt_model.py:GPTModel`，而是`mindspeed_llm/core/models/gpt/gpt_model.py:GPTModel`。这点有点奇怪，但是调试结果确实是这样的，如果要修改模型定义，记得需要修改的是`mindspeed_llm/core/models/gpt/gpt_model.py:GPTModel`。

在GPTModel中，我们可以看到每个网络层的定义，以及forward函数。这里细节比较多，有需求可以查看[链接](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/mindspeed_llm/core/models/gpt/gpt_model.py#L32)

在train函数里面则包含了主训练循环，这里细节比较多，有需求可以参考[链接](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/mindspeed_llm/training/training.py#L515)


## 调试

如果训练过程中出现错误，可以参考mindspeed里的[方法](https://fandengdong.github.io/llm/mindspeed/get_started_with_mindspeed/#%E8%B0%83%E8%AF%95mindspeed%E8%AE%AD%E7%BB%83)来进行调试。


总的来说，使用mindspeed-llm的来训练模型还是比较方便的。注意，对于多卡训练，loss信息只会在mindspeed-llm只在最后一个节点打印输出。

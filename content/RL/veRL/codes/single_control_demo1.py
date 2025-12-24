import functools
from typing import List, Any, Callable

# === Step 0: 定义装饰器和调度模式 ===
MAGIC_ATTR = "_verl_registered"

class Dispatch:
    DP_COMPUTE_PROTO = "dp_compute_proto"
    ONE_TO_ALL = "one_to_all"

def register(dispatch_mode=Dispatch.ONE_TO_ALL):
    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        setattr(inner, MAGIC_ATTR, {"dispatch_mode": dispatch_mode})
        return inner
    return decorator

# === Step 1: 模拟 DataProto（简化为 list）===
class DataProto:
    def __init__(self, data: List[Any]):
        self.data = data

    def chunk(self, n_chunks: int) -> List["DataProto"]:
        """简单按块切分，不足补 None（模拟 auto-padding）"""
        length = len(self.data)
        chunk_size = (length + n_chunks - 1) // n_chunks
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, length)
            chunk_data = self.data[start:end]
            # 补齐长度（可选）
            if len(chunk_data) < chunk_size and i == n_chunks - 1:
                chunk_data += [None] * (chunk_size - len(chunk_data))
            chunks.append(DataProto(chunk_data))
        return chunks

    def __repr__(self):
        return f"DataProto({self.data})"

# 合并多个 DataProto
def merge_dataprotos(chunks: List[DataProto]) -> DataProto:
    all_data = []
    for chunk in chunks:
        all_data.extend([x for x in chunk.data if x is not None])
    return DataProto(all_data)

# === Step 2: Worker 类（用户定义）===
class MyWorker:
    def __init__(self, rank: int):
        self.rank = rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def process(self, input_proto: DataProto):
        print(f"[Worker {self.rank}] received: {input_proto}")
        # 模拟处理：每个元素乘以 rank+1
        result_data = [x * (self.rank + 1) if x is not None else None for x in input_proto.data]
        return DataProto(result_data)

# === Step 3: WorkerGroup（自动绑定方法）===
class WorkerGroup:
    def __init__(self, workers: List[MyWorker]):
        self.workers = workers
        self.world_size = len(workers)
        self._bind_methods()

    def _bind_methods(self):
        # 扫描 MyWorker 中所有带 @register 的方法
        for attr_name in dir(MyWorker):
            method = getattr(MyWorker, attr_name)
            if hasattr(method, MAGIC_ATTR):
                attrs = getattr(method, MAGIC_ATTR)
                dispatch_mode = attrs["dispatch_mode"]

                # 简化：只支持 DP_COMPUTE_PROTO 和 ONE_TO_ALL
                if dispatch_mode == Dispatch.DP_COMPUTE_PROTO:
                    dispatch_fn = self._dispatch_dp
                    collect_fn = self._collect_dp
                elif dispatch_mode == Dispatch.ONE_TO_ALL:
                    dispatch_fn = self._dispatch_one_to_all
                    collect_fn = self._collect_all_to_list
                else:
                    raise ValueError(f"Unsupported dispatch mode: {dispatch_mode}")

                # 动态生成绑定方法
                def make_bound_method(method_name, dispatch_fn, collect_fn):
                    def bound_method(self, *args, **kwargs):
                        # 1. 分发输入
                        dispatched_args_list = dispatch_fn(args, kwargs)
                        # 2. 执行（本地模拟，实际应是远程调用）
                        results = []
                        for i, (d_args, d_kwargs) in enumerate(dispatched_args_list):
                            res = getattr(self.workers[i], method_name)(*d_args, **d_kwargs)
                            results.append(res)
                        # 3. 收集结果
                        return collect_fn(results)
                    return bound_method

                bound_func = make_bound_method(attr_name, dispatch_fn, collect_fn)
                setattr(self, attr_name, bound_func.__get__(self, WorkerGroup))

    def _dispatch_dp(self, args, kwargs):
        # 假设第一个参数是 DataProto
        input_proto = args[0]
        chunks = input_proto.chunk(self.world_size)
        dispatched = []
        for i in range(self.world_size):
            new_args = (chunks[i],) + args[1:]
            dispatched.append((new_args, kwargs))
        return dispatched

    def _collect_dp(self, results: List[DataProto]):
        return merge_dataprotos(results)

    def _dispatch_one_to_all(self, args, kwargs):
        return [(args, kwargs) for _ in range(self.world_size)]

    def _collect_all_to_list(self, results):
        return results

# === Step 4: 使用示例 ===
if __name__ == "__main__":
    # 创建 3 个 worker（模拟 3 个 GPU 或节点）
    workers = [MyWorker(rank=i) for i in range(3)]
    wg = WorkerGroup(workers)

    # 准备输入数据
    input_data = DataProto([10, 20, 30, 40, 50])  # 5 个数
    print("Input:", input_data)

    # 调用 process —— 看起来像单进程调用！
    output = wg.process(input_data)

    print("Output:", output)

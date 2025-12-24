import ray
import functools
from typing import List, Any, Optional, Dict, Tuple, Union
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# ==============================
# 1. 基础定义（不变）
# ==============================

MAGIC_ATTR = "_verl_registered"

class Dispatch:
    DP_COMPUTE_PROTO = "dp_compute_proto"

def register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO):
    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        setattr(inner, MAGIC_ATTR, {"dispatch_mode": dispatch_mode})
        return inner
    return decorator

class DataProto:
    def __init__(self, data: List[Any]):
        self.data = data

    def chunk(self, n_chunks: int) -> List["DataProto"]:
        if n_chunks == 0:
            return []
        length = len(self.data)
        if length == 0:
            return [DataProto([]) for _ in range(n_chunks)]
        chunk_size = (length + n_chunks - 1) // n_chunks
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, length)
            chunk_data = self.data[start:end]
            if i == n_chunks - 1 and len(chunk_data) < chunk_size:
                chunk_data += [None] * (chunk_size - len(chunk_data))
            chunks.append(DataProto(chunk_data))
        return chunks

    def __repr__(self):
        return f"DataProto({[x for x in self.data if x is not None]})"

def merge_dataprotos(chunks: List[DataProto]) -> DataProto:
    all_data = []
    for chunk in chunks:
        all_data.extend([x for x in chunk.data if x is not None])
    return DataProto(all_data)

# ==============================
# 2. Worker 类（稍作扩展，支持 config 和 role）
# ==============================

@ray.remote(num_gpus=1)
class MyWorker:
    def __init__(self, rank: int, config: Optional[Dict] = None, role: str = "default"):
        self.rank = rank
        self.config = config or {}
        self.role = role
        print(f"[MyWorker] Initialized on rank={rank}, role={role}, config_keys={list(config.keys()) if config else []}")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def process(self, input_proto: DataProto):
        result_data = [
            x * (self.rank + 1) if x is not None else None
            for x in input_proto.data
        ]
        return DataProto(result_data)

# ==============================
# 3. 新增：RayClassWithInitArgs（延迟实例化模板）
# ==============================

class RayClassWithInitArgs:
    """
    封装一个可远程实例化的类及其固定初始化参数。
    每个实例化时还会传入 per-worker 参数（如 rank）。
    """

    def __init__(
        self,
        cls: ray.actor.ActorClass,  # 已经 @ray.remote 的类
        init_kwargs: Optional[Dict[str, Any]] = None,
        role: str = "default"
    ):
        self.cls = cls
        self.init_kwargs = init_kwargs or {}
        self.role = role

    def create_actor_on_node(self, node_id: str, rank: int, num_gpus: int = 1):
        """
        在指定 node 上创建一个 actor，注入 rank。
        """
        scheduling_strategy = NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False
        )
        # 合并固定参数 + per-worker 参数
        final_kwargs = {**self.init_kwargs, "rank": rank, "role": self.role}
        return self.cls.options(
            scheduling_strategy=scheduling_strategy,
            num_gpus=num_gpus
        ).remote(**final_kwargs)

# ==============================
# 4. 更新：RayResourcePool（支持从模板创建）
# ==============================

class RayResourcePool:
    def __init__(self, nnodes: int, n_gpus_per_node: int):
        self.nnodes = nnodes
        self.n_gpus_per_node = n_gpus_per_node
        self.total_workers = nnodes * n_gpus_per_node

        # 获取节点列表
        cluster_resources = ray.cluster_resources()
        node_ids = sorted([k for k in cluster_resources if k.startswith("node:")])

        if len(node_ids) < nnodes:
            raise RuntimeError(f"Requested {nnodes} nodes, only {len(node_ids)} available.")

        self.target_nodes = node_ids[:nnodes]

    def create_actors_from_template(self, template: RayClassWithInitArgs) -> List[ray.actor.ActorHandle]:
        """
        使用 RayClassWithInitArgs 模板创建所有 actors。
        自动分配 rank 并绑定到节点。
        """
        actors = []
        rank = 0
        for node_id in self.target_nodes:
            for _ in range(self.n_gpus_per_node):
                actor = template.create_actor_on_node(
                    node_id=node_id,
                    rank=rank,
                    num_gpus=1
                )
                actors.append(actor)
                rank += 1
        return actors

# ==============================
# 5. RayWorkerGroup（不变）
# ==============================

class RayWorkerGroup:
    def __init__(self, worker_actors: List[ray.actor.ActorHandle]):
        self.workers = worker_actors
        self.world_size = len(worker_actors)
        self._bind_methods()

    def _bind_methods(self):
        original_cls = MyWorker.__ray_original_class__
        for attr_name in dir(original_cls):
            method = getattr(original_cls, attr_name)
            if hasattr(method, MAGIC_ATTR):
                attrs = getattr(method, MAGIC_ATTR)
                dispatch_mode = attrs["dispatch_mode"]

                if dispatch_mode == Dispatch.DP_COMPUTE_PROTO:
                    dispatch_fn = self._dispatch_dp
                    collect_fn = self._collect_dp
                else:
                    raise NotImplementedError(f"Unsupported dispatch mode: {dispatch_mode}")

                def make_bound_method(method_name, dispatch_fn, collect_fn):
                    def bound_method(self, *args, **kwargs):
                        dispatched = dispatch_fn(args, kwargs)
                        futures = [
                            getattr(self.workers[i], method_name).remote(*d_args, **d_kwargs)
                            for i, (d_args, d_kwargs) in enumerate(dispatched)
                        ]
                        results = ray.get(futures)
                        return collect_fn(results)
                    return bound_method

                bound_func = make_bound_method(attr_name, dispatch_fn, collect_fn)
                setattr(self, attr_name, bound_func.__get__(self, RayWorkerGroup))

    def _dispatch_dp(self, args, kwargs):
        input_proto = args[0]
        chunks = input_proto.chunk(self.world_size)
        return [((chunks[i],) + args[1:], kwargs) for i in range(self.world_size)]

    def _collect_dp(self, results: List[DataProto]):
        return merge_dataprotos(results)

# ==============================
# 6. 主程序：使用完整 pipeline
# ==============================

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # 模拟配置
    config = {
        "model": {"hidden_size": 768},
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 2  # 改为你实际 GPU 数量（如 2, 4）
        }
    }

    try:
        resource_pool = RayResourcePool(
            nnodes=config["trainer"]["nnodes"],
            n_gpus_per_node=config["trainer"]["n_gpus_per_node"]
        )
    except RuntimeError as e:
        print("⚠️ 资源不足，降级为本地所有 GPU")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus == 0:
            print("❌ No GPU found. This demo requires at least 1 GPU.")
            ray.shutdown()
            exit(1)
        # 手动创建 actors（不绑定节点）
        template = RayClassWithInitArgs(
            cls=MyWorker,
            init_kwargs={"config": config},
            role="actor"
        )
        worker_actors = [
            template.cls.remote(rank=i, config=config, role="actor")
            for i in range(total_gpus)
        ]
        wg = RayWorkerGroup(worker_actors)
        total_workers = total_gpus
    else:
        # ✅ 标准流程：使用 RayClassWithInitArgs + ResourcePool
        ray_cls_with_init = RayClassWithInitArgs(
            cls=MyWorker,
            init_kwargs={"config": config},
            role="actor"
        )
        worker_actors = resource_pool.create_actors_from_template(ray_cls_with_init)
        wg = RayWorkerGroup(worker_actors)
        total_workers = resource_pool.total_workers

    # 测试
    input_proto = DataProto(list(range(1, total_workers * 3 + 1)))
    print(f"\n>>> Input ({len(input_proto.data)} items): {input_proto}")
    output = wg.process(input_proto)
    print(f">>> Output: {output}\n")

    ray.shutdown()

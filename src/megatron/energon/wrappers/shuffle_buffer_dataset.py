# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.rng import WorkerRng
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset
from megatron.energon.wrappers.buffer import SavableSampleBuffer

T_sample = TypeVar("T_sample")


class ShuffleBufferDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """Shuffle buffer for the dataset."""

    size: int #缓冲区容量，控制打乱力度。越大 ↔ 相邻样本在原始数据中距离越远，但内存占用越高
    _worker_rng: WorkerRng
    _active_buffer: SavableSampleBuffer[T_sample] #SavableSampleBuffer，底层是 _buffer: List[样本] + _restore_keys: List[恢复标记] 双列表

    _savable_fields = ("_active_buffer", "_worker_rng")

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        size: int,
        *,
        worker_config: WorkerConfig,
    ):
        """Create a shuffle buffer for the dataset."""
        super().__init__(dataset, worker_config=worker_config)
        self.size = size #Shuffle 缓冲区的大小。使用蓄水池随机采样算法（Reservoir Sampling）来打乱数据顺序，size决定蓄水池的容量。
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._worker_rng = WorkerRng(self.worker_config)
        self._active_buffer = SavableSampleBuffer(self.dataset, worker_config=self.worker_config)#设置SavableSampleBuffer的dataset

    def len_worker(self, worker_idx: int | None = None) -> int:
        return self.dataset.len_worker(worker_idx) #获取当前worker进程处理的样本量

    def __iter__(self) -> Iterator[T_sample]:
        self._active_buffer.worker_start() #断点恢复时重建 buffer 中的样本
        it = iter(self._active_buffer.append_iter()) #获取进行append的迭代器，每 next() 从 inner dataset 读一条，直接append到buffer中
        while True:
            if self._active_buffer.len_worker() >= self.size: #_active_buffer.len_worker是当前缓冲区内的样本数，如果大于等于size，则从缓冲区随机取出一个样本yield，并从缓冲区pop。
                pop_idx = self._worker_rng.randbelow(self._active_buffer.len_worker()) #在缓冲区内随机取一个样本idx
                yield self._active_buffer.pop(pop_idx) #yield并pop
            else: #否则还需要继续从 inner dataset 读取填满缓冲区
                try:
                    next(it)
                except StopIteration: #耗尽了就结束
                    break
        while self._active_buffer.len_worker() > 0: #处理剩余样本，如果缓冲区还有样本，就继续yield直到缓冲区为空
            pop_idx = self._worker_rng.randbelow(self._active_buffer.len_worker())
            yield self._active_buffer.pop(pop_idx)

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        return self._active_buffer.restore_sample(restore_key)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "size": self.size,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"ShuffleBufferDataset(size={self.size}, dataset={self.dataset})"

# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Iterator, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class EpochizeDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """ #内部数据集是无限的（RepeatDataset(repeats=None)），next(_active_iter) 永远不会抛 StopIteration。但 EpochizeDataset 自己决定 yield 多少条就停
    Uses the base dataset, and creates one epoch, which has length samples. Keeps the underlying
    dataset iterator alive over epochs (i.e. if it is an infinite dataset, it will keep the state).
    Repeats the underlying dataset if the iterator is exhausted.
    """

    length: int  #length 是每个DP rank 在这个 epoch 需要 yield 的总数据量，不包括多个 DP rank 的总和
    _active_iter: Optional[Iterator[T_sample]] #内部数据集的迭代器，创建一次，跨 “epoch” 复用
    _offset: int #当前 epoch 的游标（0 ~ local_length-1）

    _savable_fields = ("_offset",)

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        length: int,
        worker_config: WorkerConfig,
    ):
        """
        Create the epochized dataset.

        Args:
            dataset: The source dataset (possibly infinite)
            length: Number of samples to iterate before iteration stops (i.e. one epoch). When
                iteration continues, the original dataset iterator is resumed and does only restart
                if exhausted.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.length = length
        self._active_iter = None

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._offset = 0

    def __iter__(self) -> Iterator[T_sample]:
        # Compute the local length for this worker, i.e. all worker's lengths sum up to the total

        if self.worker_config.num_workers <= 1: #计算当前 worker需要处理 的 local_length
            local_length = self.length
        else:
            local_length = self.length // self.worker_config.num_workers #num_workers 是单个 rank 内部的 DataLoader worker 数，不包括别的 DP rank
            if self.worker_config.rank_worker_id() < self.length % self.worker_config.num_workers: # 余数分配，前几个 worker 多一条
                local_length += 1

        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "EpochizeDataset.epoch_start",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "offset": self._offset,
                    "local_length": local_length,
                    "length": self.length,
                }
            )

        offset_range = list(range(self._offset, local_length)) #计算要 yield 的范围：[_offset, local_length)

        # Only iterate if there are samples to iterate
        if len(offset_range) > 0:
            if self._active_iter is None: #首次创建迭代器，后续复用
                self._active_iter = iter(self.dataset)

            for idx in offset_range:#迭代[_offset, local_length)这些样本，然后迭代结束（假装达到了迭代尾部，人为设置epoch范围）。下次iter的时候继续用_active_iter，读取下一个epoch数据（不同的数据）
                self._offset = (idx + 1) % local_length # 推进并自动回绕
                try:
                    sample = next(self._active_iter)
                except StopIteration:
                    break
                yield sample

        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "EpochizeDataset.epoch_end",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "offset": self._offset,
                    "local_length": local_length,
                    "length": self.length,
                }
            )

    def len_worker(self, worker_idx: int | None = None) -> int:
        if worker_idx is None:
            self.worker_config.assert_worker()
            worker_idx = self.worker_config.rank_worker_id()
        if self.worker_config.num_workers <= 1:
            assert worker_idx == 0
            return self.length
        else:
            local_length = self.length // self.worker_config.num_workers
            if worker_idx < self.length % self.worker_config.num_workers:
                local_length += 1
            return local_length

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "length": self.length,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"EpochizeDataset(length={self.length}, dataset={self.dataset})"

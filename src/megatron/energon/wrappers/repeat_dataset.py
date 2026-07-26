# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Any, Dict, Generic, Iterator, Optional, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class RepeatDataset(BaseWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset repeats the inner dataset indefinitely or a specific number of repeats."""

    repeats: Optional[Union[int, float]] #重复次数。None = 无限重复；int = 精确次数；float = 最后一次只遍历部分数据（如 2.5 表示完整遍历 2 次 + 第 3 次遍历前 50% 数据）
    _repetition: int #当前已完成的重复轮次
    _index: int #当前轮次中已遍历的样本索引

    _savable_fields = ("_repetition", "_index")

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        *,
        repeats: Optional[Union[int, float]] = None,
        restart: bool = True,
        worker_config: WorkerConfig,
    ):
        """Construct a RepeatDataset. #重复数据集逻辑

        Args:
            dataset: The input dataset to repeat.
            repeats: Number of repeats, `None` for indefinitely repeating.
            restart: If true, restart the underlying dataset after iterating once through the
                repeats if repeats is set to an integer, but still stop iterating.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset, worker_config=worker_config)
        self.repeats = repeats
        self.restart = restart

        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._repetition = 0
        self._index = 0

    def len_worker(self, worker_idx: int | None = None) -> int:
        if self.repeats is None:
            return self.dataset.len_worker(worker_idx)
        return int(self.dataset.len_worker(worker_idx) * self.repeats)

    def __iter__(self) -> Iterator[T_sample]:
        assert self.repeats is not None or self.dataset.worker_has_samples(), (
            "Cannot repeat empty dataset indefinitely"
        )

        # TODO: There is a small difference in the total sum of samples (across ranks) * repeats
        # and the sum(len_worker() for all workers across ranks).
        # This is due to the fact that the number of samples is not exactly divisible by the number of workers.

        # The dataset length is the size for the current rank. Need to divide by the number of workers
        ds_len = self.dataset.len_worker() #确定当前worker进程需要遍历的这个数据集的样本数量

        while self.repeats is None or self._repetition < self.repeats:
            if self.repeats is not None and self._repetition == math.floor(self.repeats):
                # Last iteration, adjust the number of samples
                fraction = self.repeats - math.floor(self.repeats)     #记录小数部分
                stop_after = math.floor(ds_len * fraction)  #记录不完整遍历的样本数量
                if self._index >= stop_after:
                    # We restored an index and it is already past the stop_after
                    break
            else: #完整遍历，设置stop_after=None，表示遍历完整个数据集
                stop_after = None

            for sample in self.dataset: #遍历数据集，yield sample，直到完成当前轮次的遍历（完整遍历或不完整遍历）
                self._index += 1
                yield sample
                if stop_after is not None and self._index >= stop_after:
                    break

            if self.worker_config.should_log(level=2):
                self.worker_config.worker_log(
                    {
                        "t": "RepeatDataset.repeat",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "offset": self._repetition,
                        "repeats": self.repeats,
                    }
                )
            self._repetition += 1 #更新重复轮次
            self._index = 0 #重置当前轮次的遍历索引

        if self.restart:
            self._repetition = 0
        else:
            # No more repeats
            self._repetition = math.ceil(self.repeats)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "repeats": self.repeats,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"RepeatDataset(repeats={self.repeats}, dataset={self.dataset})"

# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Iterator, List, Sequence, Tuple, TypeVar

import torch

from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.rng import WorkerRng
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


class BlendDataset(BaseWrapperDataset[T_sample, T_sample]):
    """
    This dataset wrapper blends multiple iterable datasets together give a weighting.
    The datasets may be infinite. This dataset is always infinite.
    """

    datasets: List[SavableDataset[T_sample]] #从 dataset_weights 拆出的纯数据集列表
    weights: Tuple[float, ...] #从 dataset_weights 拆出的权重元组
    dataset_weights: Sequence[Tuple[SavableDataset[T_sample], float]] #[(子数据集, 权重), ...] 列表
    exhausted: List[bool] #[bool] 标记各子数据集是否已耗尽
    _worker_rng: WorkerRng

    _savable_fields = ("exhausted", "_worker_rng")

    def __init__(
        self,
        *dataset_weights: Tuple[SavableDataset[T_sample], float],
        worker_config: WorkerConfig,
    ):
        """Construct a BlendDataset.

        Args:
            dataset_weights: Each argument should be a tuple of (dataset, weight) with a weight
                between 0 and 1. The output samples are sampled from the input datasets with the
                given probabilities.
            worker_config: Configuration for the workers.
        """
        # datasets = [dataset for dataset, _weight in dataset_weights]
        self.datasets, self.weights = zip(*dataset_weights)

        super().__init__(self.datasets, worker_config=worker_config)

        self.dataset_weights = dataset_weights
        self.reset_state_own()

    def reset_state_own(self) -> None:
        self._worker_rng = WorkerRng(self.worker_config)
        self.exhausted = [False] * len(self.weights)

    def len_worker(self, worker_idx: int | None = None) -> int: #确定当前worker在所有子数据集中的样本量
        # Give the number of samples in inner datasets, disregarding the weight
        return sum(dataset.len_worker(worker_idx) for dataset in self.datasets)

    def __iter__(self) -> Iterator[T_sample]:
        assert self.worker_has_samples(), "Cannot blend all empty datasets"

        # Create a list of datasets and their weights, but
        # set the weight to 0 if the dataset has no samples on this worker.

        dataset_iters = []
        weights = []
        for idx, (dataset, weight) in enumerate(self.dataset_weights):
            assert weight > 0, "All blending weights must be > 0"

            if dataset.worker_has_samples(): #有样本:
                dataset_iters.append(iter(dataset)) #创建迭代器，将worker对这个数据集的迭代器加入到dataset_iters里
                weights.append(weight) #保留这个数据集的采样权重，加入到weights里
            else: #没有样本，标记exhausted=True，记为数据集用完了
                dataset_iters.append(None)
                weights.append(0)
                self.exhausted[idx] = True

        weights = torch.tensor(weights, dtype=torch.float32)
        if weights.sum() == 0:
            raise RuntimeError(
                "There is a worker with no samples in any of the blended datasets. "
                "This can happen if you have a lot of workers and your dataset is too small. "
                "Currently this case is not supported."
            )

        # Some may already be exhausted on this worker when restoring a state.
        for idx, exhausted in enumerate(self.exhausted):
            if exhausted: #对于已经用完的数据集，将其权重设为0
                weights[idx] = 0
                dataset_iters[idx] = None

        while True:
            ds_idx = self._worker_rng.choice_idx(probs=weights) #按权重概率随机选子数据集

            if dataset_iters[ds_idx] is None:
                if all(dataset_iter is None for dataset_iter in dataset_iters):
                    break
                continue
            try:
                sample = next(dataset_iters[ds_idx]) #从选定的子数据集的迭代器中取出一个样本
            except StopIteration: #对于迭代器耗尽的情况
                dataset_iters[ds_idx] = None #子数据集迭代器标记为None
                weights[ds_idx] = 0 #这个子数据集的权重设为0
                self.exhausted[ds_idx] = True #标记这个子数据集已经用完了
                if all(dataset_iter is None for dataset_iter in dataset_iters):#所有子数据集都用完就结束
                    break
            else: #正确情况：取出样本
                yield add_sample_restore_key(sample, ds_idx, src=self) #对sample添加一些其他信息，比如数据来自哪个数据集等

        self.exhausted = [False] * len(self.dataset_weights) #重置 exhausted 标记，为下一次迭代做准备

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset_weights": [
                (dataset.config(), weight) for dataset, weight in self.dataset_weights
            ],
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"BlendDataset(dataset_weights={self.dataset_weights})"

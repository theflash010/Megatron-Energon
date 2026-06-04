# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Literal, Optional, Union

from megatron.energon.dataset_config import get_dataset_from_config
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseCoreDatasetFactory
from megatron.energon.flavors.webdataset.structs import DatasetSubset
from megatron.energon.metadataset.loader_interface import (
    DatasetBlendMode,
    DatasetLoaderInterface,
    LoadedDataset,
    LoadedDatasetList,
)
from megatron.energon.worker import WorkerConfig


@edataclass
class DatasetLoader(DatasetLoaderInterface):
    """Loads a dataset from a path."""     #只加载一个数据集的datasetloader类型（其他类型，比如Metadataset加载多个数据集）

    path: Union[str, EPath]
    split_part: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: Optional[int] = 1
    dataset_config: Optional[str] = None
    split_config: Optional[str] = None

    def post_initialize(self, mds_path: Optional[EPath] = None):
        pass

    def get_dataset( #返回单个数据集
        self,
        *,
        training: bool,
        split_part: Optional[str] = None,
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs: Optional[int] = 1,
        split_config: Optional[str] = None,
        dataset_config: Optional[str] = None,
        subset: Optional[DatasetSubset] = None,
        **kwargs,
    ) -> BaseCoreDatasetFactory:
        """
        Args:
            training: If true, apply training randomization.
            split_part: Default split part to use.
            worker_config: Worker configuration.
            shuffle_buffer_size: Size of the sample shuffle buffer (before task encoding).
            subflavors: Subflavors to use, might be overridden by inner datasets.
            shuffle_over_epochs: Shuffle the dataset over this many epochs.
            subset: If specified, the inner dataset(s) will be subsetted.
            **kwargs: Additional arguments to the dataset constructor.

        Returns:
            The loaded dataset
        """
        if self.split_part is not None:
            split_part = self.split_part
        if split_part is None:
            raise ValueError("Missing split part")
        if self.subflavors is not None:
            subflavors = {**self.subflavors, **(subflavors or {})}
        if split_config is None:
            split_config = self.split_config
        if dataset_config is None:
            dataset_config = self.dataset_config
        return get_dataset_from_config(
            self.path,
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavors=subflavors,
            dataset_config=dataset_config,
            split_config=split_config,
            shuffle_over_epochs=shuffle_over_epochs,
            subset=subset,
            **kwargs,
        )

    def get_datasets( #返回多个数据集，套一层 LoadedDatasetList（当然这里的DatasetLoader只加载一个数据集，所以LoadedDatasetList里面只有一个数据集）
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: Optional[int] = 1,
        subset: Optional[DatasetSubset] = None,
        **kwargs,
    ) -> LoadedDatasetList:
        return LoadedDatasetList(
            blend_mode=DatasetBlendMode.NONE,
            datasets=[
                LoadedDataset(
                    dataset=self.get_dataset(
                        training=training,
                        split_part=split_part,
                        worker_config=worker_config,
                        subflavors=subflavors,
                        shuffle_over_epochs=shuffle_over_epochs_multiplier,
                        subset=subset,
                        **kwargs,
                    ),
                    weight=None,
                )
            ],
        )

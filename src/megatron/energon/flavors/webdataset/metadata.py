# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
from enum import Enum
from typing import List, Set

import braceexpand

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.eyaml import load_yaml, load_yaml_json
from megatron.energon.flavors.webdataset.config import (
    INDEX_SQLITE_FILENAME,
    INFO_JSON_FILENAME,
    INFO_YAML_FILENAME,
    MAIN_FOLDER_NAME,
)
from megatron.energon.flavors.webdataset.empty_dataset_error import EmptyDatasetError
from megatron.energon.flavors.webdataset.structs import (
    ShardInfo,
    WebdatasetInfo,
    WebdatasetSplits,
)
from megatron.energon.typed_converter import JsonParser


class EnergonDatasetType(Enum):
    METADATASET = "metadataset"
    WEBDATASET = "webdataset"
    JSONL = "jsonl"
    FILESYSTEM = "filesystem"
    INVALID = "invalid"


@edataclass
class WebdatasetMeta:
    """Class for getting metadata from a webdataset."""

    sample_excludes: Set[str]
    shards: List[ShardInfo] # 该 split 的文件列表（只有training）
    split_part_files: List[str] # 该 split 的文件列表（只有training）
    info_shard_files: List[str] # 所有 shard 文件列表（training+val）

    @staticmethod
    def from_config(
        path: EPath,
        *,
        split_part: str,
        split_config: str | None = None,
    ) -> "WebdatasetMeta":
        """
        Loads the metadata for a webdataset, i.e. the shards and sample excludes.

        Args:
            split_part: Which part to load (e.g. 'train', 'val', 'test').
            split_config: Config file to use for shard split definitions.
        """
        if split_config is None: #设置默认 split 配置文件名
            split_config = "split.yaml"

        parser = JsonParser(strict=True)
        info_object = get_dataset_info(path) # 解析 .info.json → WebdatasetInfo，包含每个 shard 的样本数

        info = parser.raw_to_typed(
            info_object,
            WebdatasetInfo,
        )
        try: #解析 split 配置文件 → WebdatasetSplits，包括split_parts（key=split 名（"train"/"val"/"test"），value=该 split 包含的文件名列表（可含花括号通配符，如 "shard_{000..099}.tar"）），exclude（要排除的条目列表）
            splits = parser.raw_to_typed(
                load_yaml_json(path / MAIN_FOLDER_NAME / split_config),
                WebdatasetSplits,
            )
        except FileNotFoundError:
            if split_config == "split.yaml":
                # Try split.json instead
                splits = parser.raw_to_typed(
                    load_yaml_json(path / MAIN_FOLDER_NAME / "split.json"),
                    WebdatasetSplits,
                )
            else:
                raise
        assert split_part in splits.split_parts, f"Invalid split part: {split_part!r}"
        split_excludes = {
            excluded
            for excluded in splits.exclude
            for excluded in braceexpand.braceexpand(excluded)
        } #展开排除列表

        all_split_part_files = [ #展开当前 split 的文件名列表
            name
            for name in splits.split_parts[split_part]  # 取当前 split 的文件名（可能含通配符）
            for name in braceexpand.braceexpand(name) # 展开通配
        ]

        split_part_files = [name for name in all_split_part_files if name not in split_excludes] #过滤掉 exclude 的shard文件
        if len(split_part_files) == 0:
            raise EmptyDatasetError(f"No shards found in split part {split_part!r}")
        return WebdatasetMeta(
            sample_excludes={excluded for excluded in split_excludes if "/" in excluded}, #筛选出排除数据中shard名/样本索引 的格式
            shards=[
                ShardInfo(
                    name=name,
                    path=path / name,
                    count=info.shard_counts[name],
                )
                for name in split_part_files
            ],# 只构建当前 split 的 shard
            split_part_files=all_split_part_files, ## 含排除项的完整文件列表
            info_shard_files=list(info.shard_counts.keys()), ## 所有 split 的 shard
        )


def get_info_shard_files(path: EPath) -> List[str]:
    """Use this if you don't need the full metadata for split parts, but just the shard files."""
    parser = JsonParser(strict=True)
    info = parser.raw_to_typed(
        get_dataset_info(path),
        WebdatasetInfo,
    )
    return list(info.shard_counts.keys())


def get_dataset_info(path: EPath) -> dict:
    """Given the path to an energon webdataset that contains a .nv-meta folder,
    return the dataset info as a dict.
    """

    info_config = path / MAIN_FOLDER_NAME / INFO_JSON_FILENAME
    # YAML for backwards compatibility
    yaml_info_config = path / MAIN_FOLDER_NAME / ".info.yaml"

    if info_config.is_file():
        with info_config.open("r") as rf:
            return json.load(rf)
    elif yaml_info_config.is_file():
        return load_yaml(yaml_info_config.read_bytes())
    else:
        raise ValueError(f"No info config file found at {info_config} or {yaml_info_config}")


def check_dataset_info_present(path: EPath) -> bool:
    """Given the path to an energon webdataset that contains a .nv-meta folder,
    return True if the dataset info is present, False otherwise.
    """
    return (path / MAIN_FOLDER_NAME / INFO_JSON_FILENAME).is_file() or (
        path / MAIN_FOLDER_NAME / INFO_YAML_FILENAME
    ).is_file()


def get_dataset_type(path: EPath) -> EnergonDatasetType:
    """Get the type of the dataset at the given path.

    Args:
        path: The path to the dataset as specified by the user.

    Returns:
        The type of the dataset.
    """
    metadata_db = path / MAIN_FOLDER_NAME / INDEX_SQLITE_FILENAME

    if path.is_file():
        if path.name.endswith(".jsonl"):
            return EnergonDatasetType.JSONL
        elif path.name.endswith(".yaml"):
            return EnergonDatasetType.METADATASET #yaml后缀的配置文件（比如混合数据集）
        else:
            return EnergonDatasetType.INVALID
    elif check_dataset_info_present(path):
        return EnergonDatasetType.WEBDATASET
    elif metadata_db.is_file():
        # There is an sqlite, but no .info.json or .info.yaml,
        # so it's a filesystem dataset
        return EnergonDatasetType.FILESYSTEM
    else:
        return EnergonDatasetType.INVALID

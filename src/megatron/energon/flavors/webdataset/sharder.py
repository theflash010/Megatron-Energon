# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from itertools import zip_longest
from typing import Generator, Optional, Sequence, Tuple, Union

import numpy as np

from megatron.energon.flavors.webdataset.structs import DatasetSubset, ShardInfo
from megatron.energon.worker import WorkerConfig


class Sharder:
    @staticmethod
    def _split_shard(
        start_offset: int,
        end_offset: int,
        max_samples_per_sequence: Optional[int],
    ) -> Tuple[int, ...]:
        """Splits a shard into multiple slices of max_samples_per_sequence (more or less).
        Returns the starting index of each slice (excluding the end_offset)."""
        if (
            max_samples_per_sequence is not None
            and end_offset - start_offset > max_samples_per_sequence * 1.5 #超过阈值对shard（也就是单个tar文件）进行切分
        ):
            # Split the shard into slices of max_samples_per_sequence (more or less)
            slice_count = max(round((end_offset - start_offset) / max_samples_per_sequence), 1) #按照每个sequence的样本数量为max_samples_per_sequence进行切分，一共有slice_count个sequence
            samples_per_sequence = (end_offset - start_offset) / slice_count #每个sequence的样本数量
            # Note this must include the end offset as well, so slice_count + 1 steps
            return tuple(
                start_offset + int(slice * samples_per_sequence) for slice in range(slice_count) #值返回每个sequence的起始样本索引，不包括这个shard的末尾边界
            )
        else:
            return (start_offset,) #不切分

    @classmethod
    def _split_shards(
        cls,
        shard_cumsums: np.ndarray, #每个 shard 的起止偏移 [0, shard1_end, shard2_end, ...]
        offsets: Sequence[int], #worker 的样本范围 [start, end, ...]
        *,
        max_samples_per_sequence: Optional[int], #每个 slice 最大样本数
    ) -> Generator[Sequence[int], None, None]:
        """
        Splits the shards into multiple lists based on the offsets. The first offset is the start
        of the first shard emitted, the last offset is the beginning of the last shard emitted.
        (i.e. number of slice sequences emitted is `len(offsets) - 1`).

        Args:
            shard_cumsums: The source shard offsets
            offsets: The offsets to samples to get shards for (must be strictly increasing)
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                  will be sequential).

        Returns:
            A list of starting offsets for each slice (including the end offset)
        """
        # Find shard idx for start
        start_index = np.searchsorted(shard_cumsums, offsets[0], side="right") - 1

        for start_offset, end_offset in zip(offsets, offsets[1:]):
            # Find shard idx for end
            end_index = start_index
            while end_index + 1 < len(shard_cumsums) and end_offset > shard_cumsums[end_index + 1]: #不断训练直到end_offset位于end_index这个shard内
                end_index += 1
            if start_index == end_index: #起始index相同说明这个worker的样本都在同一个shard内
                yield (
                    *cls._split_shard( #获取当前shard(包含样本范围是start_offset, end_offset)在切分为sequence之后的起始样本索引
                        start_offset=start_offset,
                        end_offset=end_offset,
                        max_samples_per_sequence=max_samples_per_sequence,
                    ),
                    end_offset, #添加shard的末尾边界
                )
            else:#起始index不相同说明这个worker的样本跨越在多个shard内
                # Middle is the original shards, start and end get an offset/length
                yield (
                    *(
                        cls._split_shard(#这里处理当前worker包含的第一个shard，获取其在切分为sequence之后的起始样本索引
                            start_offset=start_offset,
                            end_offset=shard_cumsums[start_index + 1],
                            max_samples_per_sequence=max_samples_per_sequence,
                        )
                        if shard_cumsums[start_index + 1] > start_offset
                        else ()
                    ),
                    *(
                        offset
                        for inner_shard_start, inner_shard_end in zip(
                            shard_cumsums[start_index + 1 : end_index],
                            shard_cumsums[start_index + 2 : end_index + 1],
                        )
                        for offset in cls._split_shard( #这里处理当前worker包含的中间shard，获取其在切分为sequence之后的起始样本索引
                            start_offset=inner_shard_start,
                            end_offset=inner_shard_end,
                            max_samples_per_sequence=max_samples_per_sequence,
                        )
                    ),
                    *cls._split_shard( #这里处理当前worker包含的最后一个shard（不一定是完整的shard，可能是shard的一部分），获取其在切分为sequence之后的起始样本索引
                        start_offset=shard_cumsums[end_index],
                        end_offset=end_offset,
                        max_samples_per_sequence=max_samples_per_sequence,
                    ),
                    end_offset, #添加worker处理的末尾边界样本下标
                )
            start_index = end_index #更新起始index为当前处理的末尾index，方便下一个worker的处理

    @classmethod
    def _split_slices(
        cls,
        offsets: Sequence[int],
        *,
        max_samples_per_sequence: Optional[int],
    ) -> Generator[Sequence[int], None, None]:
        """
        Splits the offsets into approximately `max_samples_per_sequence` sized slices. Each sequence
        of slices includes the end of that sequence.

        Args:
            offsets: The offsets to samples to get shards for (must be strictly increasing)
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                  will be sequential).

        Returns:
            A list of offsets for each slice sequence.
        """
        for start, end in zip(offsets[:-1], offsets[1:]):
            yield (
                *cls._split_shard(
                    start_offset=start,
                    end_offset=end,
                    max_samples_per_sequence=max_samples_per_sequence,
                ),
                end,
            )

    @classmethod
    def _generalized_bit_reversal(
        cls, length_or_indices: Union[int, Sequence[int]]
    ) -> Sequence[int]:
        """This function creates a permutation of given length.
        The sequence is created by a recursive divide and interleave algorithm
        to ensure a balanced distribution across ranks.
        It corresponds to a generalized bit reversal permutation, which - for lengths
        of power of two - is the reversed binary representation of the original indices.

        For example for 16 indices, the sequence is:
            [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]

        Visual illustration:
            Step|0|1|2|3|4|5|6|7|8|9|A|B|C|D|E|F|
                |-------------------------------|
               0|X| | | | | | | | | | | | | | | |
               1|X| | | | | | | |X| | | | | | | |
               2|X| | | |X| | | |X| | | | | | | |
               3|X| | | |X| | | |X| | | |X| | | |
               4|X| |X| |X| | | |X| | | |X| | | |
               5|X| |X| |X| | | |X| |X| |X| | | |
               6|X| |X| |X| |X| |X| |X| |X| | | |
               7|X| |X| |X| |X| |X| |X| |X| |X| |
               8|X|X|X| |X| |X| |X| |X| |X| |X| |
               9|X|X|X| |X| |X| |X|X|X| |X| |X| |
              10|X|X|X| |X|X|X| |X|X|X| |X| |X| |
              11|X|X|X| |X|X|X| |X|X|X| |X|X|X| |
              12|X|X|X|X|X|X|X| |X|X|X| |X|X|X| |
              13|X|X|X|X|X|X|X| |X|X|X|X|X|X|X| |
              14|X|X|X|X|X|X|X|X|X|X|X|X|X|X|X| |
              15|X|X|X|X|X|X|X|X|X|X|X|X|X|X|X|X|
        """

        if isinstance(length_or_indices, int):
            indices = list(range(length_or_indices))
        else:
            indices = length_or_indices

        if len(indices) <= 2:
            return indices
        mid = len(indices) // 2
        left = indices[:mid]
        right = indices[mid:]

        left_result = cls._generalized_bit_reversal(left)
        right_result = cls._generalized_bit_reversal(right)

        # Interleave the results
        zipped = zip_longest(left_result, right_result)
        result = [item for sublist in zipped for item in sublist if item is not None]
        return result

    @classmethod
    def split_samples_to_workers(
        cls,
        start_samples: int,
        end_samples: int,
        worker_config: WorkerConfig,
        *,
        rotation_offset: int = 0,
    ) -> Sequence[int]:
        # We split the total number of samples into the number of global workers across all ranks.
        # Note that the global number of workers intentionally stays the same if you
        # divide the number of ranks by N, and multiply the number of workers per rank by N.
        # This allows to reproduce the same global batches with a different number of ranks.
        total_samples = end_samples - start_samples #samples数量

        num_workers = max(1, worker_config.num_workers) #每个dp rank的worker数量

        global_workers = num_workers * worker_config.world_size #global_workers 是所有dp rank的worker总数

        min_samples_per_worker = int(total_samples / global_workers) #均分的话，每个worker的样本数量（不包括余数）
        num_workers_with_more_samples = total_samples % global_workers #需要多分配一个样本的worker数量

        # We are going to compute the samples assigned to each worker on the current rank.
        # This is done in multiple steps.
        # Some of these steps could be collapsed into one, but we keep them separate for clarity:
        # 1. Compute the number of samples per global worker (rotated by rotation_offset,
        #    typically given by previous datasets).
        # 2. Permute the nuber of samples per global worker by a generalized bit reversal sequence
        # 3. Given the sample counts, compute the start and end indices for each global worker
        # 4. Extract the local worker sample assignments for the current rank.
        # 5. Split the shards based on the start and end indices.

        # 1. Let's compute it globally for all workers first
        num_samples_per_global_worker = []
        for global_worker_idx in range(global_workers):
            if (
                global_worker_idx - rotation_offset + global_workers
            ) % global_workers < num_workers_with_more_samples:
                # This worker gets one more sample 这次轮到需要多加一个样本的worker
                num_samples_per_global_worker.append(min_samples_per_worker + 1)
            else:
                # This worker gets the minimum number of samples 不需要添加样本的worker
                num_samples_per_global_worker.append(min_samples_per_worker)

        # 2. Permute the number of samples per global worker 位反转优化，不会对offset造成影响，只会改变分配给不同worker的样本数量，如果没有余数就无所谓
        worker_bitrev_seq = cls._generalized_bit_reversal(global_workers)

        # The worker_bitrev_seq is the order in which any remainder samples shall
        # be assigned to workers.
        # That means, the x-axis (array index) is the remainder sample index
        # and the y-axis (value) is the global worker index.
        # So we map the y (value) to the old global worker index from the linear sequence.

        new_num_samples_per_global_worker = [-1] * global_workers
        for old_worker_idx, new_worker_idx in enumerate(worker_bitrev_seq):
            new_num_samples_per_global_worker[new_worker_idx] = num_samples_per_global_worker[
                old_worker_idx
            ]

        num_samples_per_global_worker = new_num_samples_per_global_worker

        # 3. Compute the global worker sample start and end indices
        global_worker_sample_split_offsets = [start_samples] #计算所有worker的样本起始offset
        cur_offset = start_samples
        for global_worker_idx in range(global_workers):
            cur_offset += num_samples_per_global_worker[global_worker_idx]
            global_worker_sample_split_offsets.append(cur_offset)

        # 4. Now we extract the local rank's worker ranges 提取当前dp rank 的 local worker 的样本起始offset
        local_worker_sample_split_offsets = global_worker_sample_split_offsets[
            worker_config.rank * num_workers : (worker_config.rank + 1) * num_workers + 1
        ]

        assert len(local_worker_sample_split_offsets) == num_workers + 1, (
            "If this fails, there's a bug in the code above."
        )

        return local_worker_sample_split_offsets

    @staticmethod
    def _clean_offsets(offsets: Sequence[int]) -> Sequence[int]:
        """Removes empty offset slices, i.e. duplicates from offsets."""
        return (
            *(int(start) for start, end in zip(offsets, offsets[1:]) if start < end),
            int(offsets[-1]),
        )

    @staticmethod
    def _compute_subset(
        total_samples: int,
        subset: Optional[DatasetSubset] = None,
    ) -> tuple[int, int]:
        start_samples = 0
        end_samples = total_samples

        if subset is None:
            return start_samples, end_samples
        if subset.absolute_range is not None:
            start_samples, end_samples = subset.absolute_range
            if end_samples is None:
                end_samples = total_samples
            assert end_samples <= total_samples, (
                f"Subset samples {subset.absolute_range} {end_samples=} > {total_samples=}"
            )
            assert start_samples <= end_samples, (
                f"Subset samples {subset.absolute_range} {start_samples=} > {end_samples=}"
            )
            assert start_samples >= 0, (
                f"Subset samples {subset.absolute_range} {start_samples=} < 0"
            )
        if subset.range is not None:
            previous_total = end_samples - start_samples
            end_samples = start_samples + int(previous_total * subset.range[1])
            start_samples += int(previous_total * subset.range[0])
            assert end_samples <= total_samples, (
                f"Subset ratio {subset.range} {end_samples=} is larger than total samples {total_samples}"
            )
            assert start_samples <= end_samples, (
                f"Subset ratio {subset.range} {start_samples=} > {end_samples=}"
            )
            assert start_samples >= 0, f"Subset ratio {subset.range} {start_samples=} < 0"
        return start_samples, end_samples

    @classmethod
    def shard_workers(
        cls,
        shards: Sequence[ShardInfo],
        worker_config: WorkerConfig,
        *,
        max_samples_per_sequence: Optional[int],
        subset: Optional[DatasetSubset] = None,
        rotation_offset: int = 0,
    ) -> Sequence[Sequence[int]]:
        """
        Creates shard slices for each worker of the current rank.
        For that, the number of global samples is split across the number of global workers across all
        ranks. Then each worker gets a slice of the global samples.

        Args:
            shards: The shards to split
            worker_config: The config for the current rank and workers
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                  will be sequential).
            subset: If specified, the dataset will be subsetted to the given ratio.
            rotation_offset: The offset to use for the worker rotation.

        Returns:
            The shards for the current rank and all workers
        """
        end_samples = sum(shard.count for shard in shards) #计算shards包含的样本总数
        if subset is not None:
            start_samples, end_samples = subset.compute_subset(end_samples)
        else:
            start_samples = 0

        local_worker_sample_split_offsets = cls.split_samples_to_workers(
            start_samples,
            end_samples,
            worker_config,
            rotation_offset=rotation_offset,
        )#提取当前dp rank 的 local worker 的样本起始offset，是样本的下标，不是shards的。如：[0, 11250, 22500, 33750, 45000]

        shard_cumsums = np.cumsum([0] + [shard.count for shard in shards]) #确定每个 shard 的起止位置。 如shard_cumsums = [0, 1000, 3000, 4500]

        return tuple(
            # Filter out any empty shards for this worker
            cls._clean_offsets(offsets)#过滤空切片，移除连续重复的偏移点，如offsets = [0, 500, 500, 1000, 1000, 1500]，会输出(0, 500, 1000, 1500)
            for offsets in cls._split_shards(
                shard_cumsums,
                local_worker_sample_split_offsets,
                max_samples_per_sequence=max_samples_per_sequence,
            )#每个for循环，获取当前dp rank的一个worker处理的shard在切分为sequence（每个sequence是max_samples_per_sequence个样本）之后的起始样本索引和末尾样本索引。返回的 offsets 如：[[0, 1000], [1000, 2000], [2000, 3000]]
        )

    @classmethod
    def slice_workers(
        cls,
        total_samples: int,
        worker_config: WorkerConfig,
        *,
        max_samples_per_sequence: Optional[int],
        subset: Optional[DatasetSubset] = None,
        rotation_offset: int = 0,
    ) -> Sequence[Sequence[int]]:
        """
        Creates shard slices for each worker of the current rank.
        For that, the number of global samples is split across the number of global workers across all
        ranks. Then each worker gets a slice of the global samples.

        Args:
            total_samples: The total number of samples
            worker_config: The config for the current rank and workers
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                  will be sequential).
            subset: If specified, the dataset will be subsetted to the given ratio.
            rotation_offset: The offset to use for the worker rotation.

        Returns:
            The shards for the current rank and all workers
        """
        start_samples, end_samples = cls._compute_subset(total_samples, subset)

        local_worker_sample_split_offsets = cls.split_samples_to_workers(
            start_samples,
            end_samples,
            worker_config,
            rotation_offset=rotation_offset,
        )

        # Split the shards
        return tuple(
            # Filter out any empty shards for this worker
            cls._clean_offsets(offsets)
            for offsets in cls._split_slices(
                local_worker_sample_split_offsets,
                max_samples_per_sequence=max_samples_per_sequence,
            )
        )

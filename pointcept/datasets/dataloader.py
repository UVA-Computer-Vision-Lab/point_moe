from functools import partial
import weakref
import torch
import torch.utils.data
from collections.abc import Mapping

import pointcept.utils.comm as comm
from pointcept.datasets.utils import point_collate_fn
from pointcept.datasets import ConcatDataset
from pointcept.utils.env import set_seed


class MultiDatasetDummySampler:
    def __init__(self):
        self.dataloader = None

    def set_epoch(self, epoch):
        if comm.get_world_size() > 1:
            if isinstance(self.dataloader.dataloaders, list):
                # Iterate through multiple dataloaders
                for dataloader in self.dataloader.dataloaders:
                    if hasattr(dataloader, "sampler"):
                        dataloader.sampler.set_epoch(epoch)
            else:
                # Single dataloader case
                if hasattr(self.dataloader.dataloaders, "sampler"):
                    self.dataloader.dataloaders.sampler.set_epoch(epoch)
        return


class MultiDatasetDataloader:
    """
    Multiple Datasets Dataloader, batch data from a same dataset and mix up ratio determined by loop of each sub dataset.
    The overall length is determined by the main dataset (first) and loop of concat dataset.
    """

    def __init__(
        self,
        concat_dataset: ConcatDataset,
        batch_size_per_gpu: int,
        num_worker_per_gpu: int,
        mix_prob=0,
        batch_from_single_dataset=True,
        seed=None,
    ):
        self.mix_prob = mix_prob
        self.datasets = concat_dataset.datasets
        self.batch_from_single_dataset = batch_from_single_dataset
        self.ratios = [dataset.loop for dataset in self.datasets]
        self.batch_size_per_gpu = batch_size_per_gpu
        # reset data loop, original loop serve as ratios
        for dataset in self.datasets:
            dataset.loop = 1
        # determine union training epoch by main dataset
        self.datasets[0].loop = concat_dataset.loop
        # build sub-dataloaders
        num_workers = num_worker_per_gpu // len(self.datasets)
        if self.batch_from_single_dataset:
            self.dataloaders = []
            for dataset_id, dataset in enumerate(self.datasets):
                if comm.get_world_size() > 1:
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                else:
                    sampler = None

                init_fn = (
                    partial(
                        self._worker_init_fn,
                        dataset_id=dataset_id,
                        num_workers=num_workers,
                        num_datasets=len(self.datasets),
                        rank=comm.get_rank(),
                        seed=seed,
                    )
                    if seed is not None
                    else None
                )
                self.dataloaders.append(
                    torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size_per_gpu,
                        shuffle=(sampler is None),
                        num_workers=num_worker_per_gpu,
                        sampler=sampler,
                        collate_fn=partial(point_collate_fn, mix_prob=mix_prob),
                        pin_memory=False,
                        worker_init_fn=init_fn,
                        drop_last=True,
                        persistent_workers=True,
                    )
                )
        else:
            if comm.get_world_size() > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    concat_dataset
                )
            else:
                sampler = None

            init_fn = (
                partial(
                    self._worker_init_fn,
                    dataset_id=0,
                    num_workers=num_workers,
                    num_datasets=1,
                    rank=comm.get_rank(),
                    seed=seed,
                )
                if seed is not None
                else None
            )
            self.dataloaders = torch.utils.data.DataLoader(
                concat_dataset,
                batch_size=batch_size_per_gpu,
                shuffle=(sampler is None),
                num_workers=num_worker_per_gpu,
                sampler=sampler,
                collate_fn=partial(point_collate_fn, mix_prob=mix_prob),
                pin_memory=False,
                worker_init_fn=init_fn,
                drop_last=True,
                persistent_workers=True,
            )

        self.sampler = MultiDatasetDummySampler()
        self.sampler.dataloader = weakref.proxy(self)

    def __iter__(self):
        if self.batch_from_single_dataset:
            iterator = [iter(dataloader) for dataloader in self.dataloaders]
            total_batches = len(self)  # Ensure we stop after this many batches
            batch_count = 0

            while batch_count < total_batches:
                for i in range(len(self.ratios)):
                    for _ in range(self.ratios[i]):
                        if batch_count >= total_batches:
                            return  # Stop when the computed length is reached

                        try:
                            batch = next(iterator[i])
                        except StopIteration:
                            if i == 0:
                                return  # Stop iteration if main dataset runs out
                            else:
                                iterator[i] = iter(self.dataloaders[i])
                                batch = next(iterator[i])

                        yield batch
                        batch_count += 1  # Track batches
        else:
            for batch in self.dataloaders:
                yield batch

    def __len__(self):
        if not self.batch_from_single_dataset:
            return len(self.dataloaders)

        # Compute the total sum of dataset lengths weighted by ratios
        total_weighted_length = sum(
            len(d) * ratio for d, ratio in zip(self.dataloaders, self.ratios)
        )

        # Normalize by the total weight (sum of ratios)
        return total_weighted_length // sum(self.ratios)

    @staticmethod
    def _worker_init_fn(worker_id, num_workers, dataset_id, num_datasets, rank, seed):
        worker_seed = (
            num_workers * num_datasets * rank
            + num_workers * dataset_id
            + worker_id
            + seed
        )
        set_seed(worker_seed)

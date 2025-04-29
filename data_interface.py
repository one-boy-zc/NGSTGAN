import importlib

from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class DataInterface(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset, self.val_dataset = self.load_dataset("train"), self.load_dataset("val")
        elif stage == "test":
            self.test_dataset = self.load_dataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.kwargs["batch_size"],
                          num_workers=self.kwargs["num_workers"],
                          persistent_workers=True,
                          pin_memory=True,
                          pin_memory_device="cuda",
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.kwargs["batch_size"],
                          num_workers=self.kwargs["num_workers"],
                          persistent_workers=True,
                          pin_memory=True,
                          pin_memory_device="cuda")

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.kwargs["batch_size"],
                          num_workers=self.kwargs["num_workers"],
                          persistent_workers=True,
                          pin_memory=True,
                          pin_memory_device="cuda")

    def load_dataset(self, mode: str) -> Dataset:
        if self.kwargs["dataset_name"].find("_") == -1:
            cls_name = self.kwargs["dataset_name"].upper()
        else:
            cls_name = "".join([i.capitalize() for i in self.kwargs["dataset_name"].split("_")])

        try:
            cls = getattr(importlib.import_module("." + self.kwargs["dataset_name"],
                                                  package=__package__),
                          cls_name)

            self.kwargs.update(mode=mode)
            return cls(**self.kwargs)
        except ValueError:
            print(f"Invalid dataset name: data_scripts.{self.kwargs['dataset_name']}")

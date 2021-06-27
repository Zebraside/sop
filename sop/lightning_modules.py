import os

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.models.resnet import resnet18, resnet34


from pytorch_lightning import LightningDataModule, LightningModule

from online_triplet_loss.losses import *

from dataset import StanfordProductsOnlineDataset, Item


class StanfordProductsDataModule(LightningDataModule):
    def __init__(self, train_ann_file, dataset_dir, transforms=None, batch_size=8):
        super(StanfordProductsDataModule, self).__init__()

        items = []
        with open(train_ann_file, "r") as f:
            skip_header = True
            for line in f:
                if skip_header:
                    skip_header = False
                    continue

                img_idx, cls_idx, super_idx, name = line.split()
                items.append(Item(os.path.join(dataset_dir, name), int(cls_idx)))

        if transforms is None:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224, 224)),
            ])

        self.train = StanfordProductsOnlineDataset(items[:-3000], transforms)
        self.val = StanfordProductsOnlineDataset(items[-3000:], transforms)

        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=6)


class StanfordProductsModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.lr = 1e-3

        model = resnet34(pretrained=True)
        model.fc = torch.nn.Identity()

        self.model = model
        self.criterion = batch_hard_triplet_loss

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        imgs, positive_images, labels = batch
        imgs = torch.cat((imgs, positive_images))
        labels = torch.cat((labels, labels))
        embeddings = self.model(imgs)

        loss = self.criterion(labels, embeddings, margin=0.2, device=self.device)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, positive_images, labels = batch
        imgs = torch.cat((imgs, positive_images))
        labels = torch.cat((labels, labels))
        embeddings = self.model(imgs)

        loss = batch_hard_triplet_loss(labels, embeddings, margin=0.2, device=self.device, squared=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

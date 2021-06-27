from torchvision import transforms

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning_modules import *


def train(train_ann_path,
          images_dir,
          checkpoint_path):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    dm = StanfordProductsDataModule(train_ann_path, images_dir, trans, batch_size=32)

    stanford_model = StanfordProductsModel()

    val_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_path,
        filename='resnet_34_best_val-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
    )

    trainer = Trainer(gpus=1,
                      precision=16,
                      max_epochs=30,
                      progress_bar_refresh_rate=20,
                      callbacks=[val_checkpoint_callback],
                      auto_lr_find=False,
                      auto_scale_batch_size=False,
                      log_gpu_memory='all'
                      )

    # trainer.tune(stanford_model, datamodule=dm)
    trainer.fit(stanford_model, dm)


if __name__ == "__main__":
    train("H:\\Dataset\\Stanford_Online_Products\\Ebay_train.txt",
          "H:\\Dataset\\Stanford_Online_Products",
          "C:\\Development\\sop\\checkpoints")

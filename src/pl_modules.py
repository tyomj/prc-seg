from collections import OrderedDict
from typing import Dict, Optional

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.core.composition import Compose
from easydict import EasyDict

from src.dataset import BaseDataset
from src.utils import load_obj


class PLModel(pl.LightningModule):
    def __init__(self, prms: Dict):
        super(PLModel, self).__init__()
        print(prms)
        self.prms = prms

        # Losses
        self.losses = OrderedDict()
        for loss_name, loss in self.prms.losses.items():
            self.losses[loss_name] = load_obj(loss.class_name)()

        # Metrics
        self.metrics = OrderedDict()
        for metric_name, metric in self.prms.metrics.items():
            self.metrics[metric_name] = load_obj(metric.class_name)()
            self.metrics[metric_name].cuda()

        # Datasets
        self.train_dataset = BaseDataset(
            root=self.prms.data.root,
            subdir=self.prms.data.train_dir,
            csv_file=self.prms.data.csv_file,
            transforms=self.prms.augmentation.train.augs,
            split='train')
        self.valid_dataset = BaseDataset(
            root=self.prms.data.root,
            subdir=self.prms.data.test_dir,
            csv_file=self.prms.data.csv_file,
            transforms=self.prms.augmentation.valid.augs,
            split='val')

        # Model
        self.model = load_obj(
            self.prms.model.class_name)(**self.prms.model.params)

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   **self.prms.data.train_dl)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   **self.prms.data.valid_dl)
        return valid_loader

    def configure_optimizers(self):
        optimizer = load_obj(self.prms.optimizer.class_name)(
            self.model.parameters(), **self.prms.optimizer.params)
        scheduler = load_obj(self.prms.scheduler.class_name)(
            optimizer, **self.prms.scheduler.params)
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': self.prms.scheduler.step,
            'monitor': self.prms.scheduler.monitor
        }]

    def training_step(self, batch, index) -> Dict:
        images, mask = batch
        # get logits
        logits = self.model(images)

        # losses
        loss_dict = {}
        for loss_name, loss_func in self.losses.items():
            loss_dict[loss_name] = loss_func(logits, mask)

        # total loss
        losses = sum(loss for loss in loss_dict.values())
        loss_dict['loss'] = losses

        # log loss
        for k, v in loss_dict.items():
            self.log('train_' + k,
                     v,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)

        return {'loss': losses, 'logits': logits, 'labels': mask}

    def training_step_end(self, outputs: Dict) -> Dict:
        # metrics
        for metric_name, metric_func in self.metrics.items():
            values = metric_func(outputs['logits'], outputs['labels'])
            self.log('train_' + metric_name,
                     values,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
        del outputs['logits'], outputs['labels']
        outputs['loss'] = outputs['loss'].sum()
        return outputs

    def validation_step(self, batch, index) -> Dict:
        images, mask = batch
        # get logits
        logits = self.model(images)
        # losses
        loss_dict = {}
        for loss_name, loss_func in self.losses.items():
            loss_dict[loss_name] = loss_func(logits, mask)

        # total loss
        losses = sum(loss for loss in loss_dict.values())
        loss_dict['loss'] = losses
        # log loss
        for k, v in loss_dict.items():
            self.log('val_' + k,
                     v,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
        return {'loss': losses, 'logits': logits, 'labels': mask}

    def validation_step_end(self, outputs: Dict) -> Dict:
        for metric_name, metric_func in self.metrics.items():
            values = metric_func(outputs['logits'], outputs['labels'])
            self.log('val_' + metric_name,
                     values,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
        del outputs['logits'], outputs['labels']
        outputs['loss'] = outputs['loss'].sum()
        return outputs


class InferenceModel(PLModel):
    _default_config = {
        'augmentation': {
            'valid': {
                'augs': [{
                    'class_name': 'albumentations.Resize',
                    'params': {
                        'height': 128,
                        'width': 224,
                        'p': 1.0
                    }
                }, {
                    'class_name': 'albumentations.Normalize',
                    'params': {
                        'p': 1.0
                    }
                }, {
                    'class_name': 'albumentations.pytorch.transforms.ToTensor',
                    'params': {
                        'normalize': None
                    }
                }]
            }
        },
        'model': {
            'class_name': 'segmentation_models_pytorch.Unet',
            'params': {
                'encoder_name': 'resnet50',
                'encoder_weights': 'imagenet',
                'classes': 1,
                'activation': 'sigmoid'
            }
        },
    }

    def __init__(self, prms: Optional[Dict] = None):
        super(PLModel, self).__init__()
        if prms:
            self.prms = prms
        else:
            self.prms = self._default_config
        if not isinstance(self.prms, EasyDict):
            self.prms = EasyDict(self.prms)
        print(self.prms)
        self.trfm = self.prms.augmentation.valid.augs
        # Model
        self.model = load_obj(
            self.prms.model.class_name)(**self.prms.model.params)
        # transforms
        self.transforms = Compose(
            [load_obj(trfm.class_name)(**trfm.params) for trfm in self.trfm])

    def forward(self, image: np.ndarray):
        auged = self.transforms(image=image)['image']
        pred_mask = self.model(auged.unsqueeze(0))
        return pred_mask

    @staticmethod
    def mask2bbox(mask: np.ndarray):
        try:
            binary_mask = (mask > 0.5).astype(np.uint8)
            cntrs = cv2.findContours(binary_mask, cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_SIMPLE)[0][0][:, 0, :]
            x1, y1, x2, y2 = np.min(cntrs[:, 0]), np.min(cntrs[:, 1]), np.max(
                cntrs[:, 0]), np.max(cntrs[:, 1])
        except Exception as e:
            print(e)
            return None

        return [x1, y1, x2, y2]

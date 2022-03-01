import torch.nn as nn
import torch.nn.functional as F
import torch
from fastmri.models.varnet import SensitivityModel
import fastmri
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from model import densenet121,resnet101
from argparse import ArgumentParser
import wandb
from pytorch_lightning import Callback
import numpy as np
import wandb
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils import show
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import classification_report


class DenseNetRSS(LightningModule):
    '''
    Model for RSS combination of coils and DenseNet classifier
    '''
    def __init__(
        self,
        n_classes=2,
        drop_prob=0,
        lr=1e-4,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        make_transforms= True
    ):

        super().__init__()
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.classifier = DenseNet(
            num_classes=self.n_classes,
            number_of_coil_features=1,
            drop_rate=self.drop_prob,
        )

        self.train_labels = list()
        self.train_preds= list()
        self.val_labels= list()
        self.val_preds= list()
        self.test_preds = list()
        self.test_labels = list()

        self.transformations = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=25, translate=(0.1, 0.1)),
            ]
        )
        self.make_transformations = make_transforms

    def forward(self, input_im: torch.Tensor) -> torch.Tensor:
        input_im = torch.unsqueeze(torch.stack(list(input_im)), dim=1)
        if self.make_transformations:
            input_im = self.transformations(input_im)
        class_pred = self.classifier(input_im)
        return class_pred

    def training_step(self, batch, batch_idx):
        (
            images,
            preds,
            labels,
            loss
        ) = self._get_preds_loss_accuracy(batch)
        # Log loss and metric
        self.logger.experiment.log(
            {
                "train_loss": loss,
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )

        self.train_labels.append(labels)
        self.train_preds.append(preds)


        if batch_idx % 75 == 0:
            self.log_images(images, labels, preds, val=False)
            self.log_agg_metric_train(
                self.train_labels, self.train_preds
            )
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        images, target, mean, std, filename, dataslice, maxv, labels = batch
        class_preds = self(images)
        ce_loss = self.loss(class_preds, labels)
        preds = torch.argmax(class_preds, dim=1)
        return images, preds, labels, ce_loss

    def validation_step(self, batch, batch_idx):
        (
            images,
            preds,
            labels,
            loss
        ) = self._get_preds_loss_accuracy(batch)
        # Log loss and metric
        self.logger.experiment.log(
            {
                "validation_loss": loss,
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        self.val_preds.append(preds)
        self.val_labels.append(labels)

        if batch_idx % 100 == 0:
            self.log_images(images, labels, preds, val=True)
        return preds, {"validation_loss": loss}

    def test_step(self, batch, batch_idx):
        (
            images,
            preds,
            labels,
            loss,

        ) = self._get_preds_loss_accuracy(batch)

        self.test_labels.append(labels)
        self.test_preds.append(preds)

    def on_test_epoch_end(self):
        save_path = self.logger.save_dir
        np.save(save_path / "testlabels.npy", torch.flatten(torch.stack(self.test_labels)).cpu().numpy())
        np.save(save_path / "testpreds.npy", torch.flatten(torch.stack(self.test_preds)).cpu().numpy())
        report = classification_report(torch.flatten(torch.stack(self.test_labels)).cpu().numpy(),torch.flatten(torch.stack(self.test_preds)).cpu().numpy(), output_dict=True)
        np.save(save_path / "report.npy", report)

        self.logger.experiment.log(
            {
                "test_lesion_f1": report['1']['f1-score']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_f1": report['0']['f1-score']
            }
        )
        self.logger.experiment.log(
            {
                "test_accuracy": report['accuracy']
            }
        )
        self.logger.experiment.log(
            {
                "test_lesion_prec": report['1']['precision']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_prec": report['0']['precision']
            }
        )
        self.logger.experiment.log(
            {
                "test_lesion_rec": report['1']['recall']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_rec": report['0']['recall']
            }
        )
    def on_validation_epoch_end(self):
        self.log_agg_metric_val(self.val_labels, self.val_preds)


    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    def log_images(self, input_image, labels, preds, val=False):

        if val:
            for i in range(len(input_image)):
                self.logger.experiment.log(
                    {
                        "val classifier input": wandb.Image(
                            input_image[i],
                            caption=f"gt: {labels[i].item()}, pred: {preds[i].item()}",
                        ),
                        "global_step": self.global_step,
                        "epoch": self.current_epoch,
                    }
                )
        else:
            for i in range(len(input_image)):

                self.logger.experiment.log(
                    {
                        "classifier input": wandb.Image(
                            input_image[i],
                            caption=f"gt: {labels[i].item()}, pred: {preds[i].item()}",
                        ),
                        "global_step": self.global_step,
                        "epoch": self.current_epoch,
                    }
                )

    def log_agg_metric_train(
        self, labels, preds ):
    
        report = classification_report(torch.flatten(torch.stack(labels)).cpu().numpy(),torch.flatten(torch.stack(preds)).cpu().numpy(), output_dict=True)
        self.logger.experiment.log(
            {
                "accuracy": report['accuracy'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '1' in report.keys():

            self.logger.experiment.log(
                {
                    "lesion_f1": report['1']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
            {
                "lesion_prec": report['1']['precision'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
             )
            self.logger.experiment.log(
            {
                "lesion_rec": report['1']['recall'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
           )
        if '0' in report.keys():
            self.logger.experiment.log(
                {
                    "healthy_f1":report['0']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )

            self.logger.experiment.log(
                {
                    "healthy_prec":report['0']['precision'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        
            self.logger.experiment.log(
                {
                    "healthy_rec": report['0']['recall'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        self.train_labels = list()
        self.train_preds = list()

    def log_agg_metric_val(
        self, labels, preds
            ):
        
        report = classification_report(torch.flatten(torch.stack(labels)).cpu().numpy(),torch.flatten(torch.stack(preds)).cpu().numpy(), output_dict=True)
        self.logger.experiment.log(
            {
                "val_accuracy": report['accuracy'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '1' in report.keys():
            self.logger.experiment.log(
                {
                    "val_lesion_f1": report['1']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
            {
                "val_lesion_prec": report['1']['precision'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
         )
            self.logger.experiment.log(
            {
                "val_lesion_rec": report['1']['recall'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '0' in report.keys():
            self.logger.experiment.log(
                {
                    "val_healthy_f1": report['0']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            
            
            self.logger.experiment.log(
                {
                    "val_healthy_prec": report['0']['precision'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
           
            self.logger.experiment.log(
                {
                    "val_healthy_rec": report['0']['recall'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        self.val_labels = list()
        self.val_preds = list()

class ResNetGT(LightningModule):
    '''
    Model for JSENSE estimates of coil sensitivities and ResNet classifier
    '''
    def __init__(
        self,
        n_classes=2,
        drop_prob=0,
        lr=1e-3,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        make_transforms=False,
    ):

        super().__init__()
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay


        self.classifier = resnet101(
            num_classes=self.n_classes,
        
        )


        self.train_preds= list()
        self.train_labels= list()
        self.val_preds = list()
        self.val_labels= list()
        self.test_preds = list()
        self.test_labels = list()

        self.make_transforms = make_transforms

        if self.make_transforms:
            self.transformations = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=25, translate=(0.1, 0.1)),
                ]
            )
        

    def center_crop(self, img, tgt_size):
        _, cur_height, cur_width = img.shape

        if cur_height > tgt_size:
            yy = (cur_height - tgt_size) // 2
            img = img[:, yy : yy + tgt_size, :]
        elif cur_height < tgt_size:
            y2pad = tgt_size - cur_height
            pad1 = torch.zeros((1, y2pad // 2, cur_width)).type_as(img)
            pad2 = torch.zeros((1, y2pad - y2pad // 2, cur_width)).type_as(img)
            img = torch.cat([pad1, img, pad2], dim=-2)

        if cur_width > tgt_size:
            xx = (cur_width - tgt_size) // 2
            img = img[:, :, xx : xx + tgt_size]
        elif cur_width < tgt_size:
            w2pad = tgt_size - cur_width
            pad1 = torch.zeros((1, tgt_size, w2pad // 2)).type_as(img)
            pad2 = torch.zeros((1, tgt_size, w2pad - w2pad // 2)).type_as(img)
            img = torch.cat([pad1, img, pad2], dim=-1)

        return img

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
        ).sum(dim=0, keepdim=True)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        crop,
        sens_maps: torch.Tensor,
             ) -> torch.Tensor:
        b_s = len(masked_kspace)
        inputs = list()
        for i in range(b_s):
            sensitivity_preds = torch.view_as_real(sens_maps[i])
            input_im = self.sens_reduce(masked_kspace[i], sensitivity_preds)
    
            input_im= fastmri.complex_abs(input_im)
            input_im = self.center_crop(input_im, 320)
            input_im = (input_im - torch.mean(input_im)) / torch.std(input_im)
            inputs.append(torch.unsqueeze(input_im, dim=1))
 
        inputs = torch.cat(inputs, dim=0)
        if self.make_transforms:
            inputs = self.transformations(inputs)
        class_pred = self.classifier(inputs)

        return class_pred, input_im, (sensitivity_preds)

    def training_step(self, batch, batch_idx):
        (
            preds,
            labels,
            input_im,
            loss,
            sens,
        ) = self._get_preds_loss_accuracy(batch)
        # Log loss and metric
        self.logger.experiment.log(
            {
                "train_loss": loss,
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )

        self.train_labels.append(labels)
        self.train_preds.append(preds)

        if batch_idx % 50 == 0:
            self.log_images(batch[0], sens, input_im, labels, preds, val=False)
            self.log_agg_metric_train(
                self.train_labels, self.train_preds
            )
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        images, mask, target, filename, dataslice, maxv, crop, labels ,sensmaps = batch
        class_preds, input_im, sens = self(images, mask, crop, sensmaps)

        ce_loss = self.loss(class_preds, labels)
        preds = torch.argmax(class_preds, dim=1)

        return preds, labels, input_im, ce_loss, sens

    def validation_step(self, batch, batch_idx):
        (
            preds,
            labels,
            input_im,
            loss,
            sens,
        ) = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.logger.experiment.log(
            {
                "validation_loss": loss,
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )

        self.val_labels.append(labels)
        self.val_preds.append(preds)
        if batch_idx % 100 == 0:
            self.log_images(batch[0], sens, input_im, labels, preds, val=True)

        return preds, input_im, {"validation_loss": loss}

    def on_validation_epoch_end(self):
        self.log_agg_metric_val(
            self.val_labels, self.val_preds
        )

    def test_step(self, batch, batch_idx):
        (
            preds,
            labels,
            input_im,
            loss,
            sens,
        ) = self._get_preds_loss_accuracy(batch)

        self.test_labels.append(labels)
        self.test_preds.append(preds)

    def on_test_epoch_end(self):
        save_path = self.logger.save_dir
        np.save(save_path / "testlabels.npy", torch.flatten(torch.stack(self.test_labels)).cpu().numpy())
        np.save(save_path / "testpreds.npy", torch.flatten(torch.stack(self.test_preds)).cpu().numpy())
        report = classification_report(torch.flatten(torch.stack(self.test_labels)).cpu().numpy(),torch.flatten(torch.stack(self.test_preds)).cpu().numpy(), output_dict=True)
        np.save(save_path / "report.npy", report)

        self.logger.experiment.log(
            {
                "test_lesion_f1": report['1']['f1-score']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_f1": report['0']['f1-score']
            }
        )
        self.logger.experiment.log(
            {
                "test_accuracy": report['accuracy']
            }
        )
        self.logger.experiment.log(
            {
                "test_lesion_prec": report['1']['precision']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_prec": report['0']['precision']
            }
        )
        self.logger.experiment.log(
            {
                "test_lesion_rec": report['1']['recall']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_rec": report['0']['recall']
            }
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    def log_images(
        self, first_input, sensitivities, middle_input, labels, preds, val=False
    ):

        grid = make_grid((fastmri.complex_abs(sensitivities).permute(1, 0, 2, 3)), padding=100)
        fig = show(grid)
        if val:
            self.logger.experiment.log(
                {
                    "val sens  preds": wandb.Image(fig),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        else:
            self.logger.experiment.log(
                {
                    "sens  preds": wandb.Image(fig),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        plt.close()

       # only print the last image of batch
        image_space = fastmri.ifft2c(first_input[-1])
        image_abs = fastmri.complex_abs(image_space)
        first_image = fastmri.rss(image_abs, dim=0)
        if val:
            self.logger.experiment.log(
                {
                    "val classifier input": wandb.Image(
                        (middle_input[-1]),
                        caption=f"gt: {labels[-1].item()}, pred: {preds[-1].item()}",
                    ),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
                {
                    "val zf input": wandb.Image(first_image),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        else:
            self.logger.experiment.log(
                {
                    "classifier input": wandb.Image(
                        (middle_input[-1]),
                        caption=f"gt: {labels[-1].item()}, pred: {preds[-1].item()}",
                    ),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
                {
                    "zf input": wandb.Image(first_image),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )

    def log_agg_metric_train(
        self, labels, preds ):
    
        report = classification_report(torch.flatten(torch.stack(labels)).cpu().numpy(),torch.flatten(torch.stack(preds)).cpu().numpy(), output_dict=True)
        self.logger.experiment.log(
            {
                "accuracy": report['accuracy'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '1' in report.keys():

            self.logger.experiment.log(
                {
                    "lesion_f1": report['1']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
            {
                "lesion_prec": report['1']['precision'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
             )
            self.logger.experiment.log(
            {
                "lesion_rec": report['1']['recall'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
           )
        if '0' in report.keys():
            self.logger.experiment.log(
                {
                    "healthy_f1":report['0']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )

            self.logger.experiment.log(
                {
                    "healthy_prec":report['0']['precision'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        
            self.logger.experiment.log(
                {
                    "healthy_rec": report['0']['recall'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        self.train_labels = list()
        self.train_preds = list()

    def log_agg_metric_val(
        self, labels, preds
            ):
        

        report = classification_report(torch.flatten(torch.stack(labels)).cpu().numpy(),torch.flatten(torch.stack(preds)).cpu().numpy(), output_dict=True)
        self.logger.experiment.log(
            {
                "val_accuracy": report['accuracy'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '1' in report.keys():
            self.logger.experiment.log(
                {
                    "val_lesion_f1": report['1']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
            {
                "val_lesion_prec": report['1']['precision'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
         )
            self.logger.experiment.log(
            {
                "val_lesion_rec": report['1']['recall'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '0' in report.keys():
            self.logger.experiment.log(
                {
                    "val_healthy_f1": report['0']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            
            
            self.logger.experiment.log(
                {
                    "val_healthy_prec": report['0']['precision'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
           
            self.logger.experiment.log(
                {
                    "val_healthy_rec": report['0']['recall'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        self.val_labels = list()
        self.val_preds = list()

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        return parser

class DenseNetGT(LightningModule):
    '''
    Model for JSENSE estimates of coil sensitivities and DenseNet classifier
    '''
    def __init__(
        self,
        n_classes=2,
        drop_prob=0,
        lr=1e-3,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        make_transforms=False,
    ):

        super().__init__()
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay


        self.classifier = densenet121(
            num_classes=self.n_classes,
            number_of_coil_features=1,
            drop_rate=self.drop_prob,
        )


        self.train_preds= list()
        self.train_labels= list()
        self.val_preds = list()
        self.val_labels= list()
        self.test_preds = list()
        self.test_labels = list()

        self.make_transforms = make_transforms

        if self.make_transforms:
            self.transformations = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=25, translate=(0.1, 0.1)),
                ]
            )
        

    def center_crop(self, img, tgt_size):
        _, cur_height, cur_width = img.shape

        if cur_height > tgt_size:
            yy = (cur_height - tgt_size) // 2
            img = img[:, yy : yy + tgt_size, :]
        elif cur_height < tgt_size:
            y2pad = tgt_size - cur_height
            pad1 = torch.zeros((1, y2pad // 2, cur_width)).type_as(img)
            pad2 = torch.zeros((1, y2pad - y2pad // 2, cur_width)).type_as(img)
            img = torch.cat([pad1, img, pad2], dim=-2)

        if cur_width > tgt_size:
            xx = (cur_width - tgt_size) // 2
            img = img[:, :, xx : xx + tgt_size]
        elif cur_width < tgt_size:
            w2pad = tgt_size - cur_width
            pad1 = torch.zeros((1, tgt_size, w2pad // 2)).type_as(img)
            pad2 = torch.zeros((1, tgt_size, w2pad - w2pad // 2)).type_as(img)
            img = torch.cat([pad1, img, pad2], dim=-1)

        return img

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
        ).sum(dim=0, keepdim=True)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        crop,
        sens_maps: torch.Tensor,
             ) -> torch.Tensor:
        b_s = len(masked_kspace)
        inputs = list()
        for i in range(b_s):
            sensitivity_preds = torch.view_as_real(sens_maps[i])
            input_im = self.sens_reduce(masked_kspace[i], sensitivity_preds)
    
            input_im= fastmri.complex_abs(input_im)
            input_im = self.center_crop(input_im, 320)
            input_im = (input_im - torch.mean(input_im)) / torch.std(input_im)
            inputs.append(torch.unsqueeze(input_im, dim=1))
 
        inputs = torch.cat(inputs, dim=0)
        if self.make_transforms:
            inputs = self.transformations(inputs)
        class_pred = self.classifier(inputs)

        return class_pred, input_im, (sensitivity_preds)

    def training_step(self, batch, batch_idx):
        (
            preds,
            labels,
            input_im,
            loss,
            sens,
        ) = self._get_preds_loss_accuracy(batch)
        # Log loss and metric
        self.logger.experiment.log(
            {
                "train_loss": loss,
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )

        self.train_labels.append(labels)
        self.train_preds.append(preds)

        if batch_idx % 50 == 0:
            self.log_images(batch[0], sens, input_im, labels, preds, val=False)
            self.log_agg_metric_train(
                self.train_labels, self.train_preds
            )
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        images, mask, target, filename, dataslice, maxv, crop, labels ,sensmaps = batch
        class_preds, input_im, sens = self(images, mask, crop, sensmaps)

        ce_loss = self.loss(class_preds, labels)
        preds = torch.argmax(class_preds, dim=1)

        return preds, labels, input_im, ce_loss, sens

    def validation_step(self, batch, batch_idx):
        (
            preds,
            labels,
            input_im,
            loss,
            sens,
        ) = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.logger.experiment.log(
            {
                "validation_loss": loss,
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )

        self.val_labels.append(labels)
        self.val_preds.append(preds)
        if batch_idx % 100 == 0:
            self.log_images(batch[0], sens, input_im, labels, preds, val=True)

        return preds, input_im, {"validation_loss": loss}

    def on_validation_epoch_end(self):
        self.log_agg_metric_val(
            self.val_labels, self.val_preds
        )

    def test_step(self, batch, batch_idx):
        (
            preds,
            labels,
            input_im,
            loss,
            sens,
        ) = self._get_preds_loss_accuracy(batch)

        self.test_labels.append(labels)
        self.test_preds.append(preds)

    def on_test_epoch_end(self):
        save_path = self.logger.save_dir
        np.save(save_path / "testlabels.npy", torch.flatten(torch.stack(self.test_labels)).cpu().numpy())
        np.save(save_path / "testpreds.npy", torch.flatten(torch.stack(self.test_preds)).cpu().numpy())
        report = classification_report(torch.flatten(torch.stack(self.test_labels)).cpu().numpy(),torch.flatten(torch.stack(self.test_preds)).cpu().numpy(), output_dict=True)
        np.save(save_path / "report.npy", report)

        self.logger.experiment.log(
            {
                "test_lesion_f1": report['1']['f1-score']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_f1": report['0']['f1-score']
            }
        )
        self.logger.experiment.log(
            {
                "test_accuracy": report['accuracy']
            }
        )
        self.logger.experiment.log(
            {
                "test_lesion_prec": report['1']['precision']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_prec": report['0']['precision']
            }
        )
        self.logger.experiment.log(
            {
                "test_lesion_rec": report['1']['recall']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_rec": report['0']['recall']
            }
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    def log_images(
        self, first_input, sensitivities, middle_input, labels, preds, val=False
    ):

        grid = make_grid((fastmri.complex_abs(sensitivities).permute(1, 0, 2, 3)), padding=100)
        fig = show(grid)
        if val:
            self.logger.experiment.log(
                {
                    "val sens  preds": wandb.Image(fig),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        else:
            self.logger.experiment.log(
                {
                    "sens  preds": wandb.Image(fig),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        plt.close()

       # only print the last image of batch
        image_space = fastmri.ifft2c(first_input[-1])
        image_abs = fastmri.complex_abs(image_space)
        first_image = fastmri.rss(image_abs, dim=0)
        if val:
            self.logger.experiment.log(
                {
                    "val classifier input": wandb.Image(
                        (middle_input[-1]),
                        caption=f"gt: {labels[-1].item()}, pred: {preds[-1].item()}",
                    ),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
                {
                    "val zf input": wandb.Image(first_image),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        else:
            self.logger.experiment.log(
                {
                    "classifier input": wandb.Image(
                        (middle_input[-1]),
                        caption=f"gt: {labels[-1].item()}, pred: {preds[-1].item()}",
                    ),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
                {
                    "zf input": wandb.Image(first_image),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )

    def log_agg_metric_train(
        self, labels, preds ):
    
        report = classification_report(torch.flatten(torch.stack(labels)).cpu().numpy(),torch.flatten(torch.stack(preds)).cpu().numpy(), output_dict=True)
        self.logger.experiment.log(
            {
                "accuracy": report['accuracy'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '1' in report.keys():

            self.logger.experiment.log(
                {
                    "lesion_f1": report['1']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
            {
                "lesion_prec": report['1']['precision'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
             )
            self.logger.experiment.log(
            {
                "lesion_rec": report['1']['recall'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
           )
        if '0' in report.keys():
            self.logger.experiment.log(
                {
                    "healthy_f1":report['0']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )

            self.logger.experiment.log(
                {
                    "healthy_prec":report['0']['precision'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        
            self.logger.experiment.log(
                {
                    "healthy_rec": report['0']['recall'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        self.train_labels = list()
        self.train_preds = list()

    def log_agg_metric_val(
        self, labels, preds
            ):
        

        report = classification_report(torch.flatten(torch.stack(labels)).cpu().numpy(),torch.flatten(torch.stack(preds)).cpu().numpy(), output_dict=True)
        self.logger.experiment.log(
            {
                "val_accuracy": report['accuracy'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '1' in report.keys():
            self.logger.experiment.log(
                {
                    "val_lesion_f1": report['1']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
            {
                "val_lesion_prec": report['1']['precision'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
         )
            self.logger.experiment.log(
            {
                "val_lesion_rec": report['1']['recall'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '0' in report.keys():
            self.logger.experiment.log(
                {
                    "val_healthy_f1": report['0']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            
            
            self.logger.experiment.log(
                {
                    "val_healthy_prec": report['0']['precision'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
           
            self.logger.experiment.log(
                {
                    "val_healthy_rec": report['0']['recall'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        self.val_labels = list()
        self.val_preds = list()

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        return parser

class DenseNetPredictSens(LightningModule):
    '''
    Model for estimating coil sensitivities with U-Net and 
    classifying combined image with DenseNet classifier
    '''
    def __init__(
        self,
        n_classes=2,
        drop_prob=0,
        lr=1e-3,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        sens_pools: int = 4,
        sens_chans: int = 4,
        make_transforms=False,
    ):

        super().__init__()
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        # define sub models
        self.sensitivity_model = SensitivityModel(
            chans=self.sens_chans, num_pools=self.sens_pools
        )

        self.classifier = DenseNet(
            num_classes=self.n_classes,
            number_of_coil_features=1,
            drop_rate=self.drop_prob,
        )

        self.train_preds= list()
        self.train_labels= list()
        self.val_preds = list()
        self.val_labels= list()
        self.test_preds = list()
        self.test_labels = list()

        self.make_transforms = make_transforms

        if self.make_transforms:
            self.transformations = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=25, translate=(0.1, 0.1)),
                ]
            )
        

    def center_crop(self, img, tgt_size):
        _, cur_height, cur_width = img.shape

        if cur_height > tgt_size:
            yy = (cur_height - tgt_size) // 2
            img = img[:, yy : yy + tgt_size, :]
        elif cur_height < tgt_size:
            y2pad = tgt_size - cur_height
            pad1 = torch.zeros((1, y2pad // 2, cur_width)).type_as(img)
            pad2 = torch.zeros((1, y2pad - y2pad // 2, cur_width)).type_as(img)
            img = torch.cat([pad1, img, pad2], dim=-2)

        if cur_width > tgt_size:
            xx = (cur_width - tgt_size) // 2
            img = img[:, :, xx : xx + tgt_size]
        elif cur_width < tgt_size:
            w2pad = tgt_size - cur_width
            pad1 = torch.zeros((1, tgt_size, w2pad // 2)).type_as(img)
            pad2 = torch.zeros((1, tgt_size, w2pad - w2pad // 2)).type_as(img)
            img = torch.cat([pad1, img, pad2], dim=-1)

        return img

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )
    def forward(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor, crop
    ) -> torch.Tensor:
        b_s = len(masked_kspace)
        inputs = list()
        for i in range(b_s):
            sensitivity_preds = self.sensitivity_model(
                torch.unsqueeze(masked_kspace[i], dim=0), mask[i]
            )

            input_im = torch.squeeze(
                self.sens_reduce(masked_kspace[i], sensitivity_preds), dim=0
            )
            input_im = fastmri.complex_abs(input_im)
            input_im = self.center_crop(input_im, 320)
            input_im = (input_im - torch.mean(input_im)) / torch.std(input_im)
            inputs.append(torch.unsqueeze(input_im, dim=1))
     
        inputs = torch.cat(inputs, dim=0)
        if self.make_transforms:
            inputs = self.transformations(inputs)
        class_pred = self.classifier(inputs)

        return class_pred, input_im, sensitivity_preds

    def training_step(self, batch, batch_idx):
        (
            preds,
            labels,
            input_im,
            loss,
            sens,
        ) = self._get_preds_loss_accuracy(batch)
        # Log loss and metric
        self.logger.experiment.log(
            {
                "train_loss": loss,
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )

        self.train_labels.append(labels)
        self.train_preds.append(preds)

        if batch_idx % 50 == 0:
            self.log_images(batch[0], sens, input_im, labels, preds, val=False)
            self.log_agg_metric_train(
                self.train_labels, self.train_preds
            )
        return loss

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        images, mask, target, filename, dataslice, maxv, crop, labels = batch
        class_preds, input_im, sens = self(images, mask, crop)

        ce_loss = self.loss(class_preds, labels)
        preds = torch.argmax(class_preds, dim=1)

        return preds, labels, input_im, ce_loss, sens

    def validation_step(self, batch, batch_idx):
        (
            preds,
            labels,
            input_im,
            loss,
            sens,
        ) = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.logger.experiment.log(
            {
                "validation_loss": loss,
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )

        self.val_labels.append(labels)
        self.val_preds.append(preds)
        if batch_idx % 100 == 0:
            self.log_images(batch[0], sens, input_im, labels, preds, val=True)

        return preds, input_im, {"validation_loss": loss}

    def on_validation_epoch_end(self):
        self.log_agg_metric_val(
            self.val_labels, self.val_preds
        )

    def test_step(self, batch, batch_idx):
        (
            preds,
            labels,
            input_im,
            loss,
            sens,
        ) = self._get_preds_loss_accuracy(batch)

        self.test_labels.append(labels)
        self.test_preds.append(preds)

    def on_test_epoch_end(self):
        save_path = self.logger.save_dir
        np.save(save_path / "testlabels.npy", torch.flatten(torch.stack(self.test_labels)).cpu().numpy())
        np.save(save_path / "testpreds.npy", torch.flatten(torch.stack(self.test_preds)).cpu().numpy())
        report = classification_report(torch.flatten(torch.stack(self.test_labels)).cpu().numpy(),torch.flatten(torch.stack(self.test_preds)).cpu().numpy(), output_dict=True)
        np.save(save_path / "report.npy", report)

        self.logger.experiment.log(
            {
                "test_lesion_f1": report['1']['f1-score']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_f1": report['0']['f1-score']
            }
        )
        self.logger.experiment.log(
            {
                "test_accuracy": report['accuracy']
            }
        )
        self.logger.experiment.log(
            {
                "test_lesion_prec": report['1']['precision']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_prec": report['0']['precision']
            }
        )
        self.logger.experiment.log(
            {
                "test_lesion_rec": report['1']['recall']
            }
        )
        self.logger.experiment.log(
            {
                "test_healthy_rec": report['0']['recall']
            }
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    def log_images(
        self, first_input, sensitivities, middle_input, labels, preds, val=False
    ):

        grid = make_grid((fastmri.complex_abs(sensitivities).permute(1, 0, 2, 3)), padding=100)
        fig = show(grid)
        if val:
            self.logger.experiment.log(
                {
                    "val sens  preds": wandb.Image(fig),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        else:
            self.logger.experiment.log(
                {
                    "sens  preds": wandb.Image(fig),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        plt.close()

       # only print the last image of batch
        image_space = fastmri.ifft2c(first_input[-1])
        image_abs = fastmri.complex_abs(image_space)
        first_image = fastmri.rss(image_abs, dim=0)
        if val:
            self.logger.experiment.log(
                {
                    "val classifier input": wandb.Image(
                        (middle_input[-1]),
                        caption=f"gt: {labels[-1].item()}, pred: {preds[-1].item()}",
                    ),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
                {
                    "val zf input": wandb.Image(first_image),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        else:
            self.logger.experiment.log(
                {
                    "classifier input": wandb.Image(
                        (middle_input[-1]),
                        caption=f"gt: {labels[-1].item()}, pred: {preds[-1].item()}",
                    ),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
                {
                    "zf input": wandb.Image(first_image),
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )

    def log_agg_metric_train(
        self, labels, preds ):
    
        report = classification_report(torch.flatten(torch.stack(labels)).cpu().numpy(),torch.flatten(torch.stack(preds)).cpu().numpy(), output_dict=True)
        self.logger.experiment.log(
            {
                "accuracy": report['accuracy'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '1' in report.keys():

            self.logger.experiment.log(
                {
                    "lesion_f1": report['1']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
            {
                "lesion_prec": report['1']['precision'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
             )
            self.logger.experiment.log(
            {
                "lesion_rec": report['1']['recall'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
           )
        if '0' in report.keys():
            self.logger.experiment.log(
                {
                    "healthy_f1":report['0']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )

            self.logger.experiment.log(
                {
                    "healthy_prec":report['0']['precision'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        
            self.logger.experiment.log(
                {
                    "healthy_rec": report['0']['recall'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        self.train_labels = list()
        self.train_preds = list()

    def log_agg_metric_val(
        self, labels, preds
            ):
        

        report = classification_report(torch.flatten(torch.stack(labels)).cpu().numpy(),torch.flatten(torch.stack(preds)).cpu().numpy(), output_dict=True)
        self.logger.experiment.log(
            {
                "val_accuracy": report['accuracy'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '1' in report.keys():
            self.logger.experiment.log(
                {
                    "val_lesion_f1": report['1']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            self.logger.experiment.log(
            {
                "val_lesion_prec": report['1']['precision'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
         )
            self.logger.experiment.log(
            {
                "val_lesion_rec": report['1']['recall'],
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            }
        )
        if '0' in report.keys():
            self.logger.experiment.log(
                {
                    "val_healthy_f1": report['0']['f1-score'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
            
            
            self.logger.experiment.log(
                {
                    "val_healthy_prec": report['0']['precision'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
           
            self.logger.experiment.log(
                {
                    "val_healthy_rec": report['0']['recall'],
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                }
            )
        self.val_labels = list()
        self.val_preds = list()

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # param overwrites

        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        return parser

    

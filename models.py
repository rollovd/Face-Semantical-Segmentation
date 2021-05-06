from typing import Tuple, List, Any, Optional
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import metrics
import pytorch_lightning as pl
from dataset import FASSEG
from torch.utils.data import Dataset, DataLoader

class DeepLab(nn.Module):

    """
    Args:
      - backbone: ['resnet18', 'vgg11_bn', 'mobilenet_v3_small'],
      - aspp: use aspp module
      - num classes: num output classes
    """

    def __init__(self, backbone, aspp, num_classes):
        super(DeepLab, self).__init__()
        self.backbone = backbone
        self.init_backbone()
        self.num_classes = num_classes
        self.aspp = aspp

        if self.aspp:
            self.aspp = ASPP(self.out_features, 256, [12, 24, 36])

        self.head = DeepLabHead(self.out_features, self.num_classes)

    def init_backbone(self):

        if self.backbone == 'resnet18':
            model = models.resnet18(pretrained=True)

            self.backbone_model = nn.ModuleList([
              model.conv1, 
              model.bn1,
              model.relu,
              model.maxpool,
              model.layer1,
              model.layer2,
              model.layer3,
              model.layer4
            ]
            )

            self.out_features = model.layer4[1].bn2.num_features

        elif self.backbone == 'vgg11_bn':
            model = models.vgg11_bn(pretrained=True)

            self.backbone_model = model.features
            self.out_features = model.features[26].num_features 

        elif self.backbone == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=True)
            
            self.backbone_model = model.features
            self.out_features = model.features[-1][1].num_features 

    def _forward(self, x):
        # TODO: forward pass through the backbone
        if self.backbone == 'resnet18':
            for func in self.backbone_model:
                x = func(x)

        elif self.backbone == 'vgg11_bn':
            x = self.backbone_model(x)

        elif self.backbone == 'mobilenet_v3_small':
            x = self.backbone_model(x)

        return x

    def forward(self, inputs):

        # Model features
        x = self._forward(inputs)

        # ASPP apply
        if self.aspp:
            x = self.aspp(x)

        # Apply head
        x = self.head(x)

        # Upsample
        logits = F.interpolate(x, size=inputs.shape[-2:])

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )

class ASPPConvolve(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, rate: int) -> None:
        super(ASPPConvolve, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ASPP(nn.Module):
    """
    Args:
      - in_channels: number of input and output channels
      - num_channels: number of output channels in each intermediate "conv" block
      - atrous_rates: a list with dilation values
    """
    def __init__(self, in_channels, num_channels, atrous_rates):
        super(ASPP, self).__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels
        self.atrous_rates = atrous_rates
        self.blocks = self._fill_blocks()

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, self.num_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.Conv2d((len(self.blocks) + 1) * self.num_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def _fill_blocks(self):
        convolves = nn.ModuleList()

        # 1x1 Conv
        conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU()
        )
        convolves.append(conv)

        # Atrous rates convolutions
        for rate in self.atrous_rates:
            aspp_conv = ASPPConvolve(self.in_channels, self.num_channels, rate)
            convolves.append(aspp_conv)

        return convolves

    def forward(self, x):
        # Forward pass through the ASPP module

        initial_shape = x.shape[-2:]
        conv_prepare = [convolve(x) for convolve in self.blocks]

        # Image Pooling
        interpolation = F.interpolate(self.pool(x), size=initial_shape, mode='bilinear', align_corners=True)
        conv_prepare.append(interpolation)
  
        res = torch.cat(conv_prepare, dim=1)
        res = self.main(res)
        
        return res

class Model(pl.LightningModule):

    def __init__(
        self,
        model_name: str,
        augment_data: bool,
        optimizer: str,
        scheduler: Optional[str],
        lr: float,
        batch_size: int,
        data_path: str,
        dataset_folder: str,
        rgb_mask_path: str,
        image_size: int = 256,
    ) -> None:

        super(Model, self).__init__()
        self.num_classes = 6
        self.image_size = image_size
        self.eps = 1e-7
        self.model_name = model_name

        if self.model_name == 'deeplab_resnet':
            net = models.segmentation.deeplabv3_resnet50(pretrained=True)
            net.classifier = DeepLabHead(2048, self.num_classes)
            self.net = net

        elif self.model_name == 'deeplab_mobilenet':
            net = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
            net.classifier = DeepLabHead(960, self.num_classes)
            self.net = net

        elif self.model_name == 'deeplab_manual':
            self.net = DeepLab(backbone='mobilenet_v3_small', aspp=True, num_classes=self.num_classes)

        self.train_dataset = FASSEG(data_path=data_path, phase='Train', 
                                    dataset_folder=dataset_folder, augment=augment_data, 
                                    img_size=self.image_size, weights_rgb_path=rgb_mask_path)
        self.val_dataset = FASSEG(data_path=data_path, phase='Test', 
                                  dataset_folder=dataset_folder, augment=augment_data, 
                                  img_size=self.image_size, weights_rgb_path=rgb_mask_path)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.batch_size = batch_size

        self.color_map = torch.FloatTensor(
             [[0, 0, 0], [0, 0, 1], [0, 1, 0], 
              [0, 1, 1], [1, 0, 0], [1, 0, 1]]
        )
        self.save_hyperparameters()

    def forward(self, inputs):
        return self.net(inputs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=2, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=2, batch_size=self.batch_size, shuffle=False)

    def training_step(self, batch, batch_idx):
        img, mask = batch

        preds = self.forward(img)['out'] if self.model_name != 'deeplab_manual' else self.forward(img)
        train_loss = F.cross_entropy(preds, mask)

        self.log('train_loss', train_loss, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img)['out'] if self.model_name != 'deeplab_manual' else self.forward(img)

        intersection, union, target = metrics.calc_val_data(pred, mask, self.num_classes)

        return {'intersection': intersection, 'union': union, 'target': target, 'img': img, 'pred': pred, 'mask': mask}

    def validation_epoch_end(self, outputs):
        intersection = torch.cat([x['intersection'] for x in outputs])
        union = torch.cat([x['union'] for x in outputs])
        target = torch.cat([x['target'] for x in outputs])

        mean_iou, mean_class_acc, mean_acc = metrics.calc_val_loss(intersection, union, target, self.eps)

        log_dict = {'mean_iou': mean_iou, 'mean_class_acc': mean_class_acc, 'mean_acc': mean_acc}

        for k, v in log_dict.items():
            self.log(k, v, prog_bar=True)

        # Visualize results
        img = torch.cat([x['img'] for x in outputs]).cpu()
        pred = torch.cat([x['pred'] for x in outputs]).cpu()
        mask = torch.cat([x['mask'] for x in outputs]).cpu()

        pred_vis = self.visualize_mask(torch.argmax(pred, dim=1))
        mask_vis = self.visualize_mask(mask)

        results = torch.cat(torch.cat([img, pred_vis, mask_vis], dim=3).split(1, dim=0), dim=2)
        results_thumbnail = F.interpolate(results, scale_factor=0.25, mode='bilinear')[0]

        self.logger.experiment.add_image('results', results_thumbnail, self.current_epoch)

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        elif self.opimizer == 'SGD':
            opt = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        else:
            raise NotImplementedError("This optimzer has not been implemented.")

        if self.scheduler == 'StepLR':
            sch = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)

        elif self.scheduler == 'CyclicLR':
            sch = torch.optim.lr_scheduler.CyclicLR(opt, step_size=30)

        else:
            raise NotImplementedError('This scheduler has not been impleneted.')

        return [opt], [sch]

    def visualize_mask(self, mask):
        b, h, w = mask.shape
        mask_ = mask.view(-1)

        if self.color_map.device != mask.device:
            self.color_map = self.color_map.to(mask.device)

        mask_vis = self.color_map[mask_].view(b, h, w, 3).permute(0, 3, 1, 2).clone()

        return mask_vis
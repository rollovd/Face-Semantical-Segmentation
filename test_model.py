import argparse
from PIL import Image
from torchvision.transforms import ToTensor
import os
import numpy as np
import pickle
import metrics
from models import Model
import torch


def define_model(weights, kmeans_weights):
    model = Model.load_from_checkpoint(weights, rgb_mask_path=kmeans_weights)
    return model

def _discrete_mask(mask, kmeans_weights):

    with open(kmeans_weights, 'rb') as f:
        kmeans = pickle.load(f)

    params = mask.shape
    reshape_mask = mask.reshape(params[0] * params[1], 3)
    new_mask = kmeans.predict(reshape_mask).reshape(params[0], params[1])

    return new_mask

def define_test_images_batch(images_path, image_size):
    result = []
    to_tensor = ToTensor()

    for image_path in sorted(os.listdir(images_path)):
        image = Image.open(os.path.join(images_path, image_path))
        image = image.resize((image_size, image_size), Image.ANTIALIAS)
        image = np.asarray(image)
        image = to_tensor(image.copy())
        result.append(image)

    batch = torch.stack(result)
    return batch

def define_test_masks_batch(masks_path, image_size, kmeans_weights):
    result = []

    for mask_path in sorted(os.listdir(masks_path)):
        mask = Image.open(os.path.join(masks_path, mask_path))
        mask = mask.resize((image_size, image_size), Image.ANTIALIAS)
        mask = np.asarray(mask)
        mask = _discrete_mask(mask, kmeans_weights)
        mask = torch.from_numpy(mask.copy()).long()
        result.append(mask)

    batch = torch.stack(result)
    return batch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model-name", default='deeplab_resnet', 
                        help="Model name")
    parser.add_argument("-w", "--weights", default='', help="Model's weights path")
    parser.add_argument("-i", "--images-path", default='', help='Image test path')
    parser.add_argument("-m", "--masks-path", default='', help='Mask test path')
    parser.add_argument('-s', "--image-size", default=256, help="Image size")
    parser.add_argument('-c', "--kmeans-weights", default='', help='Mask weights')

    args = parser.parse_args()

    if args.model_name:
        model_name = args.model_name

    if args.weights:
        weights = args.weights

    if args.images_path:
        images_path = args.images_path

    if args.masks_path:
        masks_path = args.masks_path

    if args.image_size:
        image_size = args.image_size

    if args.kmeans_weights:
        kmeans_weights = args.kmeans_weights

    model = define_model(weights, kmeans_weights)
    images = define_test_images_batch(images_path, image_size)
    masks = define_test_masks_batch(masks_path, image_size, kmeans_weights)

    preds = model(images)['out']

    intersection, union, target = metrics.calc_val_data(preds, masks, 6)
    mean_iou, mean_class_acc, mean_acc = metrics.calc_val_loss(intersection, union, target, 1e-7)

    print('Mean IOU : {:.3%}'.format(mean_iou))
    print('Mean class ACC : {:.3%}'.format(mean_class_acc))
    print('Mean Acc : {:.3%}'.format(mean_acc))





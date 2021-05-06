import torch

def calculate_intersection(preds, mask, num_classes):
    """
    Logical AND between pixels
    """
    batch_size = preds.shape[0]
    intersection = torch.zeros(batch_size, num_classes)

    for i in range(num_classes):
        intersect = ((preds == i) * (mask == i)).type(torch.float).count_nonzero(dim=[1, 2])
        intersection[:, i] = intersect

    return intersection

def calculate_union(preds, mask, num_classes):
    """
    Logical OR between pixels
    """
    batch_size = preds.shape[0]
    union = torch.zeros(batch_size, num_classes)

    for i in range(num_classes):
        union_vals = ((preds == i) | (mask == i)).type(torch.float).count_nonzero(dim=[1, 2])
        union[:, i] = union_vals

    return union

def calculate_target(mask, num_classes):
    """
    Calculate quantity of distinct pixels in the image
    """
    batch_size = mask.shape[0]
    target = torch.zeros(batch_size, num_classes)

    for i in range(num_classes):
        num_of_pixels = (mask == i).sum(dim=[1, 2])
        target[:, i] = num_of_pixels
    
    return target


def calc_val_data(preds, masks, num_classes):
    """
    Calculate all metrics
    """
    preds = torch.argmax(preds, dim=1)
    
    intersection = calculate_intersection(preds, masks, num_classes) 
    union = calculate_union(preds, masks, num_classes)
    target = calculate_target(masks, num_classes)

    return intersection, union, target

def calc_val_loss(intersection, union, target, eps = 1e-7):
    """
    Calculate mean losses
    """
    mean_iou = (intersection.sum(0) / (union.sum(0) + eps)).mean() 
    mean_class_acc = (intersection.sum(0) / (target.sum(0) + eps)).mean()
    mean_acc = intersection.sum() / (target.sum() + eps) 

    return mean_iou, mean_class_acc, mean_acc
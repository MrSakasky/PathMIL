"""Image transforms used by feature encoders."""

from torchvision import transforms


def get_eval_transforms(mean, std, target_img_size: int = -1):
    operations = []
    if target_img_size > 0:
        operations.append(transforms.Resize(target_img_size))
    operations.extend((transforms.ToTensor(), transforms.Normalize(mean, std)))
    return transforms.Compose(operations)

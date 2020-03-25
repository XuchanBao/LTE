from spaghettini import quick_register
import numpy as np

from torchvision import transforms
from imgaug import augmenters as iaa
import torch


@quick_register
def resize_normalize():
    return transforms.Compose([transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,),
                                                    (0.5,))])


@quick_register
def spen_normalize():
    return transforms.Compose([transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                               ])


@quick_register
def resize_shift_augment():
    return transforms.Compose([transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Lambda(lambd=lambda x: shift_augmentation(x))])


@quick_register
def resize():
    return transforms.Compose([transforms.Resize((32, 32)),
                               transforms.ToTensor()])


@quick_register
def shift_expand():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Lambda(lambd=lambda x: shift_expand_augmentation(x))])


@quick_register
def resize_noise_normalize():
    return transforms.Compose([transforms.Resize((32, 32)),
                               SaltAndPepper(),
                               transforms.ToTensor(),
                               # transforms.Lambda(lambd=lambda img: block_left_half(img)),
                               transforms.Normalize((0.5,),
                                                    (0.5,))])


@quick_register
def one_hot_encode_transform(num_classes=10):
    return transforms.Compose([transforms.Lambda(one_hot_encoder(num_classes))])


def one_hot_encoder(num_classes):
    def one_hot_encode_with_fixed_classes(single_label):
        one_hot_label = torch.zeros((num_classes,)).float()
        one_hot_label[single_label] = 1.0

        return one_hot_label

    return one_hot_encode_with_fixed_classes


def block_left_half(img):
    num_chn, depth, width = img.shape
    img[:, :, :width // 2] = 0.0

    return img


class Blur:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.0))),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class SaltAndPepper:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.SaltAndPepper(0.85),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


def shift_augmentation(x, max_shift=2):
    bs, height, width = x.shape

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(bs, height, width)
    shifted_image[:, source_height_slice, source_width_slice] = x[:, target_height_slice, target_width_slice]

    return shifted_image.float()


def shift_expand_augmentation(x, max_shift=4):
    channels, height, width = x.shape

    # Create new image.
    expanded_img = torch.zeros(channels, height + 2 * max_shift, width + 2 * max_shift)

    # Sample shifts and slices.
    h_shift, w_shift = np.random.randint(-1 * max_shift, max_shift + 1, size=2)
    height_slice = slice(4 + h_shift, height + max_shift + h_shift)
    width_slice = slice(4 + w_shift, width + max_shift + w_shift)

    # Paste original image into the expanded image.
    expanded_img[:, height_slice, width_slice] = x

    return expanded_img

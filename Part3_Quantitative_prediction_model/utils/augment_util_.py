from torchvision import transforms as tfs
from PIL import Image
import numpy as np

augment_operat = tfs.Compose([
    tfs.RandomHorizontalFlip(0.5),
    tfs.RandomVerticalFlip(0.5),
    tfs.RandomRotation(180,expand=True),
    tfs.Resize([256,256]),
    tfs.RandomResizedCrop(256)]
)

trans = tfs.Resize(224)
def train_vit_transform(image):
    a = Image.fromarray(image.astype(np.uint8))
    image = trans(a)
    return np.array(image).astype(np.float32)

def train_multi_augment(image):
    a = Image.fromarray(image.astype(np.uint8))
    image = augment_operat(a)
    return np.array(image).astype(np.float32)

def train_multi_augment2(image,p):
    augment_func_list = [
        lambda image: (image), # default
        augment_flipud,                    # up-down
        augment_fliplr,                    # left-right
        augment_transpose,                 # transpose
    ]
    c = np.random.choice(len(augment_func_list),p=p)
    image = augment_func_list[c](image)
    return image

def augment_default(image, mask=None):
    if mask is None:
        return image
    else:
        return image, mask

def augment_flipud(image, mask=None):
    image = np.flipud(image)
    if mask is None:
        return image
    else:
        mask = np.flipud(mask)
        return image, mask

def augment_fliplr(image, mask=None):
    image = np.fliplr(image)
    if mask is None:
        return image
    else:
        mask = np.fliplr(mask)
        return image, mask

def augment_transpose(image, mask=None):
    image = np.transpose(image, (1, 0, 2))
    if mask is None:
        return image
    else:
        if len(mask.shape) == 2:
            mask = np.transpose(mask, (1, 0))
        else:
            mask = np.transpose(mask, (1, 0, 2))
        return image, mask

def augment_flipud_lr(image, mask=None):
    image = np.flipud(image)
    image = np.fliplr(image)
    if mask is None:
        return image
    else:
        mask = np.flipud(mask)
        mask = np.fliplr(mask)
        return image, mask

def augment_flipud_transpose(image, mask=None):
    if mask is None:
        image = augment_flipud(image, mask=mask)
        image = augment_transpose(image, mask=mask)
        return image
    else:
        image, mask = augment_flipud(image, mask=mask)
        image, mask = augment_transpose(image, mask=mask)
        return image, mask

def augment_fliplr_transpose(image, mask=None):
    if mask is None:
        image = augment_fliplr(image, mask=mask)
        image = augment_transpose(image, mask=mask)
        return image
    else:
        image, mask = augment_fliplr(image, mask=mask)
        image, mask = augment_transpose(image, mask=mask)
        return image, mask

def augment_flipud_lr_transpose(image, mask=None):
    if mask is None:
        image = augment_flipud(image, mask=mask)
        image = augment_fliplr(image, mask=mask)
        image = augment_transpose(image, mask=mask)
        return image
    else:
        image, mask = augment_flipud(image, mask=mask)
        image, mask = augment_fliplr(image, mask=mask)
        image, mask = augment_transpose(image, mask=mask)
        return image, mask
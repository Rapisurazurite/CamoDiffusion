import os
import albumentations
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance

from dataset.dataset_utils import boundary_modification


# several data augumentation strategies
def cv_random_flip(img, label):
    flip_flag = random.randint(0, 3)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_flag == 2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip_flag == 3:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


def random_modified(gt, iou_max=1.0, iou_min=0.8):
    iou_target = np.random.rand() * (iou_max - iou_min) + iou_min
    seg = boundary_modification.modify_boundary((np.array(gt) > 0.5).astype('uint8') * 255, iou_target=iou_target)
    return seg


# dataset for training
class PolypObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, mean=None, std=None, randomPeper=True,
                 boundary_modification=False, boundary_args={}):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        import albumentations
        self.aug_transform = albumentations.Compose([
            albumentations.RandomScale(scale_limit=0.25, p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=15, p=0.5),
            albumentations.RandomRotate90(p=0.5),
            # albumentations.ElasticTransform(p=0.5),
        ])
        self.img_transform = self.get_transform(mean, std)
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images)
        self.do_randomPeper = randomPeper
        self.do_boundary_modification = boundary_modification
        self.boundary_args = boundary_args

    def get_transform(self, mean=None, std=None):
        mean = [0.485, 0.456, 0.406] if mean is None else mean
        std = [0.229, 0.224, 0.225] if std is None else std
        transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        return transform

    def __getitem__(self, index):
        data = {}
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image_size = image.size
        # data augumentation
        image, gt = self.aug_transform(image=np.asarray(image), mask=np.asarray(gt)).values()
        image, gt = albumentations.PadIfNeeded(*image_size[::-1], border_mode=0)(image=image, mask=gt).values()
        image, gt = albumentations.RandomCrop(*image_size[::-1])(image=image, mask=gt).values()
        if self.do_boundary_modification:
            seg = random_modified(gt, **self.boundary_args)
            seg = self.gt_transform(Image.fromarray(seg))
            data['seg'] = seg

        image = colorEnhance(Image.fromarray(image))
        gt = randomPeper(gt) if self.do_randomPeper else Image.fromarray(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        data['image'] = image
        data['gt'] = gt

        return data

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset(data.Dataset):
    def __init__(self, image_root, gt_root, testsize, mean=None, std=None):
        super(test_dataset, self).__init__()
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.transform = self.get_transform(mean, std)
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, image_for_post

    def get_transform(self, mean=None, std=None):
        mean = [0.485, 0.456, 0.406] if mean is None else mean
        std = [0.229, 0.224, 0.225] if std is None else std
        transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        return transform

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in range(self.size):
            yield self.load_data()

    def __getitem__(self, item):
        image = self.rgb_loader(self.images[item])
        image_for_post = image.copy()
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[item])

        name = self.images[item].split('/')[-1]

        # This is for initial predictor.
        image_for_post = self.get_transform()(image_for_post)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return {'image': image, 'gt': gt, 'name': name, 'image_for_post': image_for_post}

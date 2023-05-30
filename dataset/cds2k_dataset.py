from PIL import Image
from torch.utils.data import dataset
import torchvision.transforms as transforms
from pathlib import Path
from glob import glob


class cds2K_dataset(dataset.Dataset):
    def __init__(self, root, testsize, mean=None, std=None):
        super().__init__()
        self.testsize = testsize
        self.root = root
        self.images = glob(str(Path(root) / '*' / 'Image' / '*.jpg'))
        self.gts = glob(str(Path(root) / '*' / 'GroundTruth' / '*.png'))
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        assert len(self.images) == len(self.gts)
        self.transform = self.get_transform(mean, std)
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

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

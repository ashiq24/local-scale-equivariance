import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms.functional as F


class MultiDigitMNIST(Dataset):
    def __init__(
            self,
            root,
            train=True,
            transform=None,
            target_transform=None,
            download=True,
            img_size=64,
            scales=None,
            concat_type='horizontal',
            image_channels=1,
            n_digits=2):
        """
        Initializes the MultiDigitMNIST dataset. This class concatenate 'n_digits' digits into a single image.
        And scale each digit to a random scale between 'scales'. As each of the digists are scaled differently 
        it creats local scaling effect, with different scaling factor at different location.

        Arguments:
        - root: Path to MNIST data
        - train: True for training set, False for test set
        - transform: Transforms to apply to the images
        - target_transform: Transforms to apply to the labels
        - download: Whether to download the dataset
        - img_size: Target square size of each digit image. 
        - scales: List of (min, max) tuples for scaling each digit
        - concat_type: 'horizontal', 'vertical', 'diagonal', or 'all'
        - image_channels: Number of channels in output image (1 or 3)
        - n_digits: Number of digits to concatenate (2 or 3)

        Example:
        -----------

        dataset = MultiDigitMNIST(
            root='./data',
            train=True,
            transform=transform,
            scales=[(0.8, 1.2), (0.7, 1.3), (0.9, 1.1)],
            concat_type='diagonal',
            image_channels=1,
            n_digits=3
        )
        """
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            transform=None,
            target_transform=target_transform,
            download=download)

        self.img_size = img_size
        self.scales = scales or [(1.0, 1.0)] * n_digits
        self.concat_type = concat_type
        self.image_channels = image_channels
        self.n_digits = n_digits

        assert len(self.scales) == n_digits, "Scales list must match n_digits"
        assert n_digits in [2, 3], "Currently supports 2 or 3 digits"
        assert concat_type in ['horizontal', 'vertical', 'diagonal', 'all']

        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def to_pil(self, img):
        """Convert tensor or numpy array to PIL Image"""
        if isinstance(img, torch.Tensor):
            return F.to_pil_image(img)
        return img

    def scale_and_pad(self, img, scale_factor):
        """Scale and center-pad an individual digit"""
        img = self.to_pil(img)
        orig_w, orig_h = img.size
        
        # Scale the image
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create padded square image
        padded = Image.new('L', (self.img_size, self.img_size), color=0)
        pad_left = (self.img_size - new_w) // 2
        pad_top = (self.img_size - new_h) // 2
        padded.paste(img, (pad_left, pad_top))
        
        return padded

    def __getitem__(self, idx, scale_list=None):
        # Get consecutive digits with wrap-around
        digits = [self.mnist[(idx + i) % len(self.mnist)] 
                 for i in range(self.n_digits)]
        imgs, labels = zip(*digits)


        processed = []
        for i, (img, scale) in enumerate(zip(imgs, self.scales)):
            # Generate deterministic random scale for reproducibility
            random.seed(idx + i)
            scale_factor = scale_list[i] if scale_list is not None else random.uniform(scale[0], scale[1])
            processed.append(self.scale_and_pad(img, scale_factor))

        # Determine concatenation type
        if self.concat_type == 'all':
            random.seed(idx)
            concat_type = random.choice(['horizontal', 'vertical', 'diagonal'])
        else:
            concat_type = self.concat_type

        # Concatenate images
        base_size = self.img_size
        if concat_type == 'horizontal':
            total_width = base_size * self.n_digits
            combined = Image.new('L', (total_width, base_size))
            for i, img in enumerate(processed):
                combined.paste(img, (i * base_size, 0))
            final_size = max(total_width, base_size)
            
        elif concat_type == 'vertical':
            total_height = base_size * self.n_digits
            combined = Image.new('L', (base_size, total_height))
            for i, img in enumerate(processed):
                combined.paste(img, (0, i * base_size))
            final_size = max(base_size, total_height)
            
        elif concat_type == 'diagonal':
            final_size = base_size * self.n_digits
            combined = Image.new('L', (final_size, final_size))
            for i, img in enumerate(processed):
                combined.paste(img, (i * base_size, i * base_size))

        # Convert to square canvas
        final_img = Image.new('L', (final_size, final_size), color=0)
        final_img.paste(combined, (
            (final_size - combined.width) // 2,
            (final_size - combined.height) // 2
        ))

        # Convert to RGB if needed
        if self.image_channels == 3:
            final_img = final_img.convert('RGB')

        # Apply transforms
        if self.transform:
            final_img = self.transform(final_img)

        # Create combined label
        combined_label = sum(label * (10 ** (self.n_digits - 1 - i)) 
                            for i, label in enumerate(labels))

        return final_img, combined_label


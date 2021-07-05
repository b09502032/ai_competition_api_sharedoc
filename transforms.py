import numbers

import PIL.Image
import timm.data
import torch
import torchvision
import torchvision.transforms.functional as F


class SquarePad(torch.nn.Module):
    def __init__(self, fill=0, padding_mode="constant", location="random"):
        super().__init__()

        if not isinstance(fill, (numbers.Number, str, tuple)):
            raise TypeError("Got inappropriate fill arg")

        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        if location != "random" and not (0 <= location and location <= 1):
            raise ValueError

        self.fill = fill
        self.padding_mode = padding_mode
        self.location = location

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be square padded.

        Returns:
            PIL Image or Tensor: Square padded image.
        """
        if isinstance(img, PIL.Image.Image):
            height = img.height
            width = img.width
        else:
            assert isinstance(img, torch.Tensor)
            assert img.ndim == 3
            height, width = img.shape[1:]

        total_padding = max(height, width) - min(height, width)

        location = self.location
        if location == "random":
            location = torch.rand(1).item()
        before = int(total_padding * location)
        after = total_padding - before

        # If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
        # (https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.pad)
        if height < width:
            padding = [0, before, 0, after]
        else:
            padding = [before, 0, after, 0]

        return F.pad(img, padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(fill={0}, padding_mode={1}, location={2})'.format(self.fill, self.padding_mode, self.location)


grayscale = torch.nn.ModuleList([torchvision.transforms.Grayscale(3)])
gaussian_blur = torch.nn.ModuleList([torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2.0))])

train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.RandomApply(grayscale, p=0.5),
        torchvision.transforms.RandomApply(gaussian_blur, p=0.5),
        torchvision.transforms.ToTensor(),
    ]
)
train_squarepad128_transform = torchvision.transforms.Compose(
    [
        SquarePad(fill=255, location='random'),
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.RandomApply(grayscale, p=0.5),
        torchvision.transforms.RandomApply(gaussian_blur, p=0.5),
        torchvision.transforms.ToTensor(),
    ]
)
inception_v3_train_transform = torchvision.transforms.Compose(
    [
        SquarePad(fill=255, location='random'),
        torchvision.transforms.Resize((299, 299)),
        torchvision.transforms.RandomApply(grayscale, p=0.5),
        torchvision.transforms.RandomApply(gaussian_blur, p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
crazy_train_transform = torchvision.transforms.Compose(
    [
        SquarePad(fill=255, location='random'),
        torchvision.transforms.Resize((128, 128)),
        timm.data.RandAugment(timm.data.rand_augment_ops()),
        torchvision.transforms.ToTensor(),
    ]
)

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((128, 128)),
])
val_squarepad128_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),  # numpy.ndarray compatibility
        SquarePad(fill=1, location=0.5),  # white is 1 in torch.Tensor
        torchvision.transforms.Resize((128, 128)),
    ]
)
inception_v3_val_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),  # numpy.ndarray compatibility
        SquarePad(fill=1, location=0.5),  # white is 1 in torch.Tensor
        torchvision.transforms.Resize((299, 299)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

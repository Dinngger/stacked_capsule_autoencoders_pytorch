from absl import flags
import torch
from torchvision import datasets, transforms
data_root = 'pytorch_data'

flags.DEFINE_integer('canvas_size', 28, 'Canvas size.')


def getMnist(train_kwargs, test_kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST(data_root, train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST(data_root, train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader

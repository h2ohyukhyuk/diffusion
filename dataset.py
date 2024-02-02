import torchvision.transforms as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from utils import DeviceDataLoader

def get_dataset(dataset_name='MNIST'):
    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize((32, 32),
                      interpolation=TF.InterpolationMode.BICUBIC,
                      antialias=True),
            #             TF.RandomHorizontalFlip(),
            TF.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
        ]
    )

    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-10":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Flowers":
        dataset = datasets.ImageFolder(root="/kaggle/input/flowers-recognition/flowers", transform=transforms)

    return dataset


def get_dataloader(dataset_name='MNIST',
                   batch_size=32,
                   pin_memory=False,
                   shuffle=True,
                   num_workers=0,
                   device="cpu"
                   ):
    dataset = get_dataset(dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            pin_memory=pin_memory,
                            shuffle=shuffle
                            ) #num_workers=num_workers,
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader
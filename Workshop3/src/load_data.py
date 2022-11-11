import torch
import torchvision
import torchvision.transforms as transforms

class CIFAR10Data():
    def __init__(self, batch_size, download=False, root='./data'):
        self.root = root # where to save data
        self.batch_size = batch_size
        self.classes = (
            'plane', 'car', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck'
        ) # CIFAR10 labels
        self.train_loader, self.test_loader = self.loaders(download)

    def loaders(self, download):
        '''
        From PyTorch:
            The output of torchvision datasets are PILImage images of range [0, 1]. 
            We transform them to Tensors of normalized range [-1, 1].
        '''
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root=self.root, train=True,
            download=download, transform=transform
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size,
            shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root=self.root, train=False,
            download=download, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size,
            shuffle=False, num_workers=2
        )

        return train_loader, test_loader
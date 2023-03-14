from torchvision import datasets, transforms


# TODO: Add datasets imagenette/birds/svhn etc etc.
def get_dataset(data_dir, im_size, c):
    ts = [] if c==3 else [transforms.Grayscale()]
    ts +=  [transforms.ToTensor(),
            transforms.Resize(im_size),
            transforms.Normalize((0.5,), (0.5,))
    ]
    transform_train = transforms.Compose(ts)

    if data_dir == "mnist":
        train_set = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif data_dir == "cifar":
        train_set = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
    else:
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )

    return train_set


from .mnist_data_module import MNISTDataModule

available_pl_modules = {
    'mnist': MNISTDataModule,
}


def get_pl_datamodule(name):
    return available_pl_modules[name]


def get_available_pl_modules():
    return list(available_pl_modules.keys())

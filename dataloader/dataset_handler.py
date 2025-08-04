from .mnist_data_module import MNISTDataModule


def get_data_module(params):
    if params.data_set == 'mnist':
        return MNISTDataModule(data_dir=params.data_dir,
                               batch_size=params.batch_size,
                               num_workers=params.num_workers,
                               val_ratio=params.val_ratio,
                               image_size=params.image_size,
                               s=params.s,
                               n_digits=params.n_digits,
                               concat_type=params.concat_type,
                               image_channels=params.image_channels,
                               image_format=params.image_format)
    else:
        raise ValueError(f'Dataset {params.data_set} not found')


def get_test_data_module(params):
    if params.data_set == 'mnist':
        return MNISTDataModule(data_dir=params.data_dir,
                               batch_size=params.batch_size,
                               num_workers=params.num_workers,
                               val_ratio=params.val_ratio,
                               image_size=params.image_size,
                               s=params.test_s,
                               n_digits=params.n_digits,
                               concat_type=params.concat_type,
                               image_channels=params.image_channels,
                               image_format=params.image_format)
    else:
        raise ValueError(f'Dataset {params.data_set} not found')

import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms as T

def create_dataset(data_path, mode, img_size, batch_size=32, shuffle=True, num_parallel_workers=1, drop_remainder=False):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path, mode)

    # define map operations
    img_transforms = [
        CV.Rescale(1.0 / 255.0, 0),
        CV.Resize(img_size, CV.Inter.BILINEAR),
        CV.Normalize([0.5], [0.5]),
        CV.HWC2CHW()
    ]
    label_transforms = [
        T.TypeCast(mindspore.int32)
    ]
    
    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=img_transforms, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=label_transforms, input_columns="label", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    if shuffle:
        mnist_ds = mnist_ds.shuffle(buffer_size=1024)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=drop_remainder)

    return mnist_ds
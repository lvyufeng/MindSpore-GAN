import os
import sys
import argparse
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
from mindspore import context
from tqdm import tqdm

sys.path.append(os.pardir)
from grad import value_and_grad
from layers import Dense
from img_utils import to_image
from amp import DynamicLossScale, NoLossScale, auto_mixed_precision, all_finite

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--amp", type=bool, default=False, help="automatic mixed precision")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
latent_dim = opt.latent_dim

if context.get_context('device_target') == 'Ascend':
    amp = True
else:
    amp = opt.amp


class Generator(nn.Cell):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [Dense(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.SequentialCell(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            Dense(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def construct(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.SequentialCell(
            Dense(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            Dense(512, 256),
            nn.LeakyReLU(0.2),
            Dense(256, 1),
            nn.Sigmoid(),
        )

    def construct(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)

        return validity

# Loss function
adversarial_loss = nn.BCELoss(reduction='mean')

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
generator.update_parameters_name('generator')
discriminator.update_parameters_name('discriminator')

if amp:
    auto_mixed_precision(generator)
    auto_mixed_precision(discriminator)
    loss_scale_D = DynamicLossScale(1024, 2, 100)
    loss_scale_G = DynamicLossScale(1024, 2, 100)
else:
    loss_scale_D = NoLossScale()
    loss_scale_G = NoLossScale()

generator.set_train()
discriminator.set_train()

# Optimizers
optimizer_G = nn.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)

# dataset

def create_dataset(data_path, mode, batch_size=32, shuffle=True, num_parallel_workers=1, drop_remainder=False):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path, mode)

    # define map operations
    transforms = [
        CV.Rescale(1.0 / 255.0, 0),
        CV.Resize(opt.img_size, CV.Inter.BILINEAR),
        CV.Normalize([0.5], [0.5]),
        CV.HWC2CHW()
    ]
    
    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=transforms, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    if shuffle:
        mnist_ds = mnist_ds.shuffle(buffer_size=1024)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=drop_remainder)

    return mnist_ds


class TrainStep(nn.Cell):
    def __init__(self):
        super().__init__()
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.loss_scale_G = loss_scale_G
        self.loss_scale_D = loss_scale_D

        self.grad_generator_fn = value_and_grad(self.generator_forward,
                                                self.optimizer_G.parameters,
                                                has_aux=True)
        self.grad_discriminator_fn = value_and_grad(self.discriminator_forward,
                                                    self.optimizer_D.parameters)

    def generator_forward(self, real_imgs, valid):
        # Sample noise as generator input
        z = ops.StandardNormal()((real_imgs.shape[0], latent_dim))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        if amp:
            g_loss = self.loss_scale_G.scale(g_loss)
        return g_loss, gen_imgs

    def discriminator_forward(self, real_imgs, gen_imgs, valid, fake):
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
        d_loss = (real_loss + fake_loss) / 2
        if amp:
            d_loss = self.loss_scale_D.scale(d_loss)
        return d_loss

    def construct(self, imgs):
        valid = ops.ones((imgs.shape[0], 1), imgs.dtype)
        fake = ops.zeros((imgs.shape[0], 1), imgs.dtype)

        (g_loss, (gen_imgs,)), g_grads = self.grad_generator_fn(imgs, valid)
        d_loss, d_grads = self.grad_discriminator_fn(imgs, gen_imgs, valid, fake)
        if amp:
            # g_loss_scale
            g_grads = self.loss_scale_G.unscale(g_grads)
            g_loss = self.loss_scale_G.unscale(g_loss)
            grads_finite_G = all_finite(g_grads)
            self.loss_scale_G.adjust(grads_finite_G)
            if grads_finite_G:
                self.optimizer_G(g_grads)
            # d_loss_scale
            d_grads = self.loss_scale_G.unscale(d_grads)
            d_loss = self.loss_scale_G.unscale(d_loss)
            grads_finite_D = all_finite(d_grads)
            self.loss_scale_D.adjust(grads_finite_D)
            if grads_finite_D:
                self.optimizer_D(d_grads)
        else:
            self.optimizer_G(g_grads)
            self.optimizer_D(d_grads)

        return g_loss, d_loss, gen_imgs

train_step = TrainStep()

dataset = create_dataset('../../dataset', 'train', opt.batch_size, num_parallel_workers=opt.n_cpu)
dataset_size = dataset.get_dataset_size()

for epoch in range(opt.n_epochs):
    t = tqdm(total=dataset_size)
    t.set_description('Epoch %i' % epoch)
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        g_loss, d_loss, gen_imgs = train_step(imgs)
        t.set_postfix(g_loss=g_loss, d_loss=d_loss)
        t.update(1)
        batches_done = epoch * dataset_size + i
        if batches_done % opt.sample_interval == 0:
            to_image(gen_imgs[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
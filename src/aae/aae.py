import os
import sys
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import ms_function
from tqdm import tqdm

sys.path.append(os.pardir)
from grad import value_and_grad
from layers import Dense
from img_utils import to_image
from dataset import create_dataset

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
latent_dim = opt.latent_dim

def reparameterization(mu, logvar):
    std = ops.exp(logvar / 2)
    sampled_z =  mnp.randn((mu.shape[0], latent_dim))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.SequentialCell(
            Dense(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )

        self.mu = Dense(512, latent_dim)
        self.logvar = Dense(512, latent_dim)

    def construct(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z

class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.SequentialCell(
            Dense(latent_dim, 512),
            nn.LeakyReLU(0.2),
            Dense(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            Dense(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def construct(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.SequentialCell(
            Dense(latent_dim, 512),
            nn.LeakyReLU(0.2),
            Dense(512, 256),
            nn.LeakyReLU(0.2),
            Dense(256, 1),
            nn.Sigmoid(),
        )

    def construct(self, z):
        validity = self.model(z)
        return validity


# Use binary cross-entropy loss
adversarial_loss = nn.BCELoss(reduction='mean')
pixelwise_loss = nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

encoder.update_parameters_name('encoder')
decoder.update_parameters_name('decoder')
discriminator.update_parameters_name('discriminator')
encoder.set_train()
decoder.set_train()
discriminator.set_train()

# Optimizers
optimizer_G = nn.Adam(
    encoder.trainable_params() + decoder.trainable_params(),
    learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_G.update_parameters_name('optim_g')
optimizer_D.update_parameters_name('optim_d')

def generator_forward(real_imgs, valid):
    encoded_imgs = encoder(real_imgs)
    decoded_imgs = decoder(encoded_imgs)

    # Loss measures generator's ability to fool the discriminator
    g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
        decoded_imgs, real_imgs
    )

    return g_loss, encoded_imgs

def discriminator_forward(encoded_imgs, valid, fake):
    # Sample noise as discriminator ground truth
    z = mnp.randn((imgs.shape[0], latent_dim))

    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(z), valid)
    fake_loss = adversarial_loss(discriminator(encoded_imgs), fake)
    d_loss = 0.5 * (real_loss + fake_loss)
    return d_loss

grad_generator_fn = value_and_grad(generator_forward,
                                   optimizer_G.parameters,
                                   has_aux=True)
grad_discriminator_fn = value_and_grad(discriminator_forward,
                                       optimizer_D.parameters)

@ms_function
def train_step(imgs):
    valid = ops.ones((imgs.shape[0], 1), mindspore.float32)
    fake = ops.zeros((imgs.shape[0], 1), mindspore.float32)

    (g_loss, (encoded_imgs,)), g_grads = grad_generator_fn(imgs, valid)
    optimizer_G(g_grads)
    d_loss, d_grads = grad_discriminator_fn(encoded_imgs, valid, fake)
    optimizer_D(d_grads)

    return g_loss, d_loss

dataset = create_dataset('../../dataset', 'train', opt.img_size, opt.batch_size, num_parallel_workers=opt.n_cpu)
dataset_size = dataset.get_dataset_size()

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = mnp.randn((n_row ** 2, latent_dim))
    gen_imgs = decoder(z)
    to_image(gen_imgs, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

for epoch in range(opt.n_epochs):
    t = tqdm(total=dataset_size)
    t.set_description('Epoch %i' % epoch)
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        g_loss, d_loss = train_step(imgs)
        t.set_postfix(g_loss=g_loss, d_loss=d_loss)
        t.update(1)
        batches_done = epoch * dataset_size + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
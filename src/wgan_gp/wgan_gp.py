import os
import sys
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
from mindspore import ms_function, context
from tqdm import tqdm

sys.path.append(os.pardir)
from grad import value_and_grad, grad
from layers import Dense
from utils import to_image

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
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
latent_dim = opt.latent_dim
n_critic = opt.n_critic

if context.get_context('device_target') == 'Ascend':
    compute_type = mindspore.float16
else:
    compute_type = mindspore.float32

class Generator(nn.Cell):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [Dense(in_feat, out_feat).to_float(compute_type)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.SequentialCell(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            Dense(1024, int(np.prod(img_shape))).to_float(compute_type),
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
            Dense(int(np.prod(img_shape)), 512).to_float(compute_type),
            nn.LeakyReLU(0.2),
            Dense(512, 256).to_float(compute_type),
            nn.LeakyReLU(0.2),
            Dense(256, 1).to_float(compute_type),
        )

    def construct(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
generator.update_parameters_name('generator')
discriminator.update_parameters_name('discriminator')
generator.set_train()
discriminator.set_train()

# Optimizers
optimizer_G = nn.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_G.update_parameters_name('optim_g')
optimizer_D.update_parameters_name('optim_d')

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

def compute_gradient_penalty(real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = ops.StandardNormal()((real_samples.shape[0], 1, 1, 1))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    
    grad_fn = grad(discriminator)
    # Get gradient w.r.t. interpolates
    (gradients,) = grad_fn(interpolates)

    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((mnp.norm(gradients, 2, axis=1) - 1) ** 2).mean()
    return gradient_penalty

def discriminator_forward(real_imgs):
    # Sample noise as generator input
    z = ops.StandardNormal()((real_imgs.shape[0], latent_dim))

    # Generate a batch of images
    fake_imgs = generator(z)

    # Real images
    real_validity = discriminator(real_imgs)
    # Fake images
    fake_validity = discriminator(fake_imgs)
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(real_imgs, fake_imgs)
    # Adversarial loss
    d_loss = -ops.reduce_mean(real_validity) + ops.reduce_mean(fake_validity) + lambda_gp * gradient_penalty
    
    return d_loss, z

def generator_forward(z):
    # Generate a batch of images
    fake_imgs = generator(z)
    # Loss measures generator's ability to fool the discriminator
    # Train on fake images
    fake_validity = discriminator(fake_imgs)
    g_loss = -ops.reduce_mean(fake_validity)

    return g_loss, fake_imgs

grad_generator_fn = value_and_grad(generator_forward,
                                   optimizer_G.parameters,
                                   has_aux=True)
grad_discriminator_fn = value_and_grad(discriminator_forward,
                                       optimizer_D.parameters,
                                       has_aux=True)

@ms_function
def train_step_d(imgs):
    (d_loss, (z,)), d_grads = grad_discriminator_fn(imgs)
    optimizer_D(d_grads)
    return d_loss, z

@ms_function
def train_step_g(z):
    (g_loss, (fake_imgs,)), g_grads = grad_generator_fn(z)
    optimizer_G(g_grads)

    return g_loss, fake_imgs

dataset = create_dataset('../../dataset', 'train', opt.batch_size, num_parallel_workers=opt.n_cpu)
dataset_size = dataset.get_dataset_size()

batches_done = 0

for epoch in range(opt.n_epochs):
    t = tqdm(total=dataset_size)
    t.set_description('Epoch %i' % epoch)
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        d_loss, z = train_step_d(imgs)
        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:
            g_loss, fake_imgs = train_step_g(z)
            if batches_done % opt.sample_interval == 0:
                to_image(fake_imgs[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += opt.n_critic
        t.set_postfix(g_loss=g_loss, d_loss=d_loss)
        t.update(1)

import os
import sys
import argparse
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from tqdm import tqdm
from mindspore import ms_function
from mindspore.common.initializer import initializer, Normal

sys.path.append(os.pardir)
from grad import value_and_grad
from layers import Dense, Upsample, Embedding, Conv2d, Dropout2d
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
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
latent_dim = opt.latent_dim

def weights_init_normal(top_cell: nn.Cell):
    for _, cell in top_cell.cells_and_names():
        classname = cell.__class__.__name__
        if classname.find("Conv") != -1:
            cell.weight.set_data(initializer(Normal(0.02), cell.weight.shape))
        elif classname.find("BatchNorm2d") != -1:
            cell.gamma.set_data(initializer(Normal(0.02, 1.0), cell.gamma.shape))
            cell.beta.set_data(initializer('zeros', cell.beta.shape))


class Generator(nn.Cell):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.SequentialCell([Dense(opt.latent_dim, 128 * self.init_size ** 2)])

        self.conv_blocks = nn.SequentialCell(
            nn.BatchNorm2d(128),
            Upsample(scale_factor=2),
            Conv2d(128, 128, 3, stride=1, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2),
            Upsample(scale_factor=2),
            Conv2d(128, 64, 3, stride=1, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2),
            Conv2d(64, opt.channels, 3, stride=1, padding=1, pad_mode='pad'),
            nn.Tanh(),
        )

    def construct(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.SequentialCell(nn.Conv2d(opt.channels, 64, 3, 2, 'pad', 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2
        self.fc = nn.SequentialCell(
            Dense(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(),
            Dense(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(),
        )
        # Upsampling
        self.up = nn.SequentialCell(Upsample(scale_factor=2), Conv2d(64, opt.channels, 3, 1, 'pad', 1))

    def construct(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.shape[0], -1))
        out = self.up(out.view(out.shape[0], 64, self.down_size, self.down_size))
        return out

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

# Initialize weights
weights_init_normal(generator)
weights_init_normal(discriminator)

def generator_forward(imgs):
    # Sample noise as generator input
    batch_size = imgs.shape[0]
    z = mnp.randn((batch_size, latent_dim))

    # Generate a batch of images
    gen_imgs = generator(z)

    # Loss measures generator's ability to fool the discriminator
    g_loss = ops.reduce_mean(ops.abs(discriminator(gen_imgs) - gen_imgs))

    return g_loss, gen_imgs

def discriminator_forward(real_imgs, gen_imgs, k):
    # Measure discriminator's ability to classify real from generated samples
    d_real = discriminator(real_imgs)
    d_fake = discriminator(gen_imgs)

    d_loss_real = ops.reduce_mean(ops.abs(d_real - real_imgs))
    d_loss_fake = ops.reduce_mean(ops.abs(d_fake - gen_imgs))
    d_loss = d_loss_real - k * d_loss_fake

    return d_loss, d_loss_real, d_loss_fake

def compute_weight_term(d_loss_real, d_loss_fake, k):
    diff = ops.reduce_mean(gamma * d_loss_real - d_loss_fake)

    # Update weight term for fake samples
    k = k + lambda_k * diff.asnumpy()
    k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

    # Update convergence metric
    M = (d_loss_real + ops.abs(diff)).asnumpy()
    return float(k), M

grad_generator_fn = value_and_grad(generator_forward,
                                   optimizer_G.parameters,
                                   has_aux=True)
grad_discriminator_fn = value_and_grad(discriminator_forward,
                                       optimizer_D.parameters,
                                       has_aux=True)

@ms_function
def train_step(imgs, k):
    (g_loss, (gen_imgs,)), g_grads = grad_generator_fn(imgs)
    optimizer_G(g_grads)
    (d_loss, (d_loss_real, d_loss_fake)), d_grads = grad_discriminator_fn(imgs, gen_imgs, k)
    optimizer_D(d_grads)

    return g_loss, d_loss, gen_imgs, d_loss_real, d_loss_fake

dataset = create_dataset('../../dataset', 'train', opt.img_size, opt.batch_size, num_parallel_workers=opt.n_cpu)
dataset_size = dataset.get_dataset_size()

# BEGAN hyper parameters
gamma = 0.75
lambda_k = 0.001
k = 0.0

for epoch in range(opt.n_epochs):
    t = tqdm(total=dataset_size)
    t.set_description('Epoch %i' % epoch)
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        g_loss, d_loss, gen_imgs, d_loss_real, d_loss_fake = train_step(imgs, k)
        k, M = compute_weight_term(d_loss_real, d_loss_fake, k)
        t.set_postfix(g_loss=g_loss, d_loss=d_loss, k=k, M=M)
        t.update(1)
        batches_done = epoch * dataset_size + i
        if batches_done % opt.sample_interval == 0:
            to_image(gen_imgs[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
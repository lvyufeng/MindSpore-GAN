import os
import sys
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ms_function
from tqdm import tqdm
from mindspore.common.initializer import initializer, Normal

sys.path.append(os.pardir)
from grad import value_and_grad
from layers import Dense, Upsample, Embedding, Conv2d, Dropout2d
from img_utils import to_image
from dataset import create_dataset

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
latent_dim = opt.latent_dim

def weights_init_normal(top_cell: nn.Cell):
    for _, cell in top_cell.cells_and_names():
        classname = cell.__class__.__name__
        if classname.find("Dense") != -1:
            cell.weight.set_data(initializer(Normal(0.02), cell.weight.shape))
        elif classname.find("BatchNorm") != -1:
            cell.gamma.set_data(initializer(Normal(0.02, 1.0), cell.gamma.shape))
            cell.beta.set_data(initializer('zeros', cell.beta.shape))

class CoupledGenerators(nn.Cell):
    def __init__(self):
        super(CoupledGenerators, self).__init__()

        self.init_size = opt.img_size // 4
        self.fc = nn.SequentialCell([Dense(opt.latent_dim, 128 * self.init_size ** 2)])

        self.shared_conv = nn.SequentialCell(
            nn.BatchNorm2d(128),
            Upsample(scale_factor=2),
            Conv2d(128, 128, 3, stride=1, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2),
            Upsample(scale_factor=2),
        )
        self.G1 = nn.SequentialCell(
            Conv2d(128, 64, 3, stride=1, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2),
            Conv2d(64, opt.channels, 3, stride=1, padding=1, pad_mode='pad'),
            nn.Tanh(),
        )
        self.G2 = nn.SequentialCell(
            Conv2d(128, 64, 3, stride=1, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2),
            Conv2d(64, opt.channels, 3, stride=1, padding=1, pad_mode='pad'),
            nn.Tanh(),
        )

    def construct(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)
        return img1, img2


class CoupledDiscriminators(nn.Cell):
    def __init__(self):
        super(CoupledDiscriminators, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [Conv2d(in_filters, out_filters, 3, 2, 'pad', 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2), Dropout2d(0.25)])
            return block

        self.shared_conv = nn.SequentialCell(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.D1 = Dense(128 * ds_size ** 2, 1)
        self.D2 = Dense(128 * ds_size ** 2, 1)

    def construct(self, img1, img2):
        # Determine validity of first image
        out = self.shared_conv(img1)
        out = out.view(out.shape[0], -1)
        validity1 = self.D1(out)
        # Determine validity of second image
        out = self.shared_conv(img2)
        out = out.view(out.shape[0], -1)
        validity2 = self.D2(out)

        return validity1, validity2


# Loss function
adversarial_loss = nn.MSELoss()

# Initialize models
coupled_generators = CoupledGenerators()
coupled_discriminators = CoupledDiscriminators()
coupled_generators.update_parameters_name('generator')
coupled_discriminators.update_parameters_name('discriminator')
coupled_generators.set_train()
coupled_discriminators.set_train()

# Initialize weights
weights_init_normal(coupled_generators)
weights_init_normal(coupled_discriminators)


# Optimizers
optimizer_G = nn.Adam(coupled_generators.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.Adam(coupled_discriminators.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_G.update_parameters_name('optim_g')
optimizer_D.update_parameters_name('optim_d')

def generator_forward(real_imgs, valid):
    # Sample noise as generator input
    z = ops.StandardNormal()((real_imgs.shape[0], latent_dim))

    # Generate a batch of images
    gen_imgs = generator(z)

    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_imgs), valid)

    return g_loss, gen_imgs

def discriminator_forward(real_imgs, gen_imgs, valid, fake):
    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
    d_loss = (real_loss + fake_loss) / 2
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

    (g_loss, (gen_imgs,)), g_grads = grad_generator_fn(imgs, valid)
    optimizer_G(g_grads)
    d_loss, d_grads = grad_discriminator_fn(imgs, gen_imgs, valid, fake)
    optimizer_D(d_grads)

    return g_loss, d_loss, gen_imgs

dataset = create_dataset('../../dataset', 'train', opt.img_size, opt.batch_size, num_parallel_workers=opt.n_cpu)
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
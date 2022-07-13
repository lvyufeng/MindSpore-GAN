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
from layers import Dense, Embedding
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
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
latent_dim = opt.latent_dim
n_classes = opt.n_classes

class Generator(nn.Cell):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [Dense(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.SequentialCell(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            Dense(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def construct(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = ops.concat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.SequentialCell(
            Dense(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            Dense(512, 512),
            nn.Dropout(1 - 0.4),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 512),
            nn.Dropout(1 - 0.4),
            nn.LeakyReLU(0.2),
            Dense(512, 1),
        )

    def construct(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = ops.concat((img.view(img.shape[0], -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

# Loss function
adversarial_loss = nn.MSELoss()

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

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = mnp.randn((n_row ** 2, opt.latent_dim))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = mindspore.Tensor(labels, mindspore.int64)
    gen_imgs = generator(z, labels)
    to_image(gen_imgs, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

def generator_forward(imgs, valid):
    batch_size = imgs.shape[0]
    z = mnp.randn((batch_size, latent_dim))
    gen_labels = mnp.randint(0, n_classes, batch_size)

    # Generate a batch of images
    gen_imgs = generator(z, gen_labels)

    # Loss measures generator's ability to fool the discriminator
    validity = discriminator(gen_imgs, gen_labels)
    g_loss = adversarial_loss(validity, valid)


    return g_loss, gen_imgs, gen_labels

def discriminator_forward(real_imgs, labels, gen_imgs, gen_labels, valid, fake):
    # Loss for real images
    validity_real = discriminator(real_imgs, labels)
    d_real_loss = adversarial_loss(validity_real, valid)

    # Loss for fake images
    validity_fake = discriminator(gen_imgs, gen_labels)
    d_fake_loss = adversarial_loss(validity_fake, fake)

    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2

    return d_loss

grad_generator_fn = value_and_grad(generator_forward,
                                   optimizer_G.parameters,
                                   has_aux=True)
grad_discriminator_fn = value_and_grad(discriminator_forward,
                                       optimizer_D.parameters)

@ms_function
def train_step(imgs, labels):
    valid = ops.ones((imgs.shape[0], 1), mindspore.float32)
    fake = ops.zeros((imgs.shape[0], 1), mindspore.float32)

    (g_loss, (gen_imgs, gen_labels)), g_grads = grad_generator_fn(imgs, valid)
    optimizer_G(g_grads)
    d_loss, d_grads = grad_discriminator_fn(imgs, labels, gen_imgs, gen_labels, valid, fake)
    optimizer_D(d_grads)

    return g_loss, d_loss

dataset = create_dataset('../../dataset', 'train', opt.img_size, opt.batch_size, num_parallel_workers=opt.n_cpu)
dataset_size = dataset.get_dataset_size()

for epoch in range(opt.n_epochs):
    t = tqdm(total=dataset_size)
    t.set_description('Epoch %i' % epoch)
    for i, (imgs, labels) in enumerate(dataset.create_tuple_iterator()):
        g_loss, d_loss = train_step(imgs, labels)
        t.set_postfix(g_loss=g_loss, d_loss=d_loss)
        t.update(1)
        batches_done = epoch * dataset_size + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
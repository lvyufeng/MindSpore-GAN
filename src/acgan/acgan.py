import os
import sys
import argparse
import numpy as np
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
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
latent_dim = opt.latent_dim
n_classes = opt.n_classes

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

        self.label_emb = Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
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

    def construct(self, noise, labels):
        gen_input = ops.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [
                Conv2d(in_filters, out_filters, 3, 2, 'pad', 1),
                nn.LeakyReLU(0.2),
                Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.SequentialCell(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.SequentialCell([Dense(128 * ds_size ** 2, 1), nn.Sigmoid()])
        self.aux_layer = nn.SequentialCell([Dense(128 * ds_size ** 2, opt.n_classes), nn.Softmax()])

    def construct(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

# Loss functions
adversarial_loss = nn.BCELoss(reduction='mean')
auxiliary_loss = nn.CrossEntropyLoss()

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
    validity, pred_label = discriminator(gen_imgs)
    g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

    return g_loss, gen_imgs, gen_labels

def discriminator_forward(real_imgs, labels, gen_imgs, gen_labels, valid, fake):
    # Loss for real images
    real_pred, real_aux = discriminator(real_imgs)
    d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

    # Loss for fake images
    fake_pred, fake_aux = discriminator(gen_imgs)
    d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2

    return d_loss, real_aux, fake_aux

def accuracy(real_aux, fake_aux, labels, gen_labels):
    # Calculate discriminator accuracy
    pred = np.concatenate([real_aux.asnumpy(), fake_aux.asnumpy()], axis=0)
    gt = np.concatenate([labels.asnumpy(), gen_labels.asnumpy()], axis=0)
    d_acc = np.mean(np.argmax(pred, axis=1) == gt)
    return d_acc

grad_generator_fn = value_and_grad(generator_forward,
                                   optimizer_G.parameters,
                                   has_aux=True)
grad_discriminator_fn = value_and_grad(discriminator_forward,
                                       optimizer_D.parameters,
                                       has_aux=True)

@ms_function
def train_step(imgs, labels):
    valid = ops.ones((imgs.shape[0], 1), mindspore.float32)
    fake = ops.zeros((imgs.shape[0], 1), mindspore.float32)

    (g_loss, (gen_imgs, gen_labels)), g_grads = grad_generator_fn(imgs, valid)
    optimizer_G(g_grads)
    (d_loss, (real_aux, fake_aux)), d_grads = grad_discriminator_fn(imgs, labels, gen_imgs, gen_labels, valid, fake)
    optimizer_D(d_grads)

    return g_loss, d_loss, gen_labels, real_aux, fake_aux

dataset = create_dataset('../../dataset', 'train', opt.img_size, opt.batch_size, num_parallel_workers=opt.n_cpu)
dataset_size = dataset.get_dataset_size()

for epoch in range(opt.n_epochs):
    t = tqdm(total=dataset_size)
    t.set_description('Epoch %i' % epoch)
    for i, (imgs, labels) in enumerate(dataset.create_tuple_iterator()):
        g_loss, d_loss, gen_labels, real_aux, fake_aux = train_step(imgs, labels)
        acc = accuracy(real_aux, fake_aux, labels, gen_labels)
        t.set_postfix(g_loss=g_loss, d_loss=d_loss, acc=f'%d%%' % (100 * acc))
        t.update(1)
        batches_done = epoch * dataset_size + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
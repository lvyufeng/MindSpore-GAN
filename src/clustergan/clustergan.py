import os
import sys
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import ms_function
from mindspore.common.initializer import initializer, Normal
from tqdm import tqdm

sys.path.append(os.pardir)
from grad import value_and_grad, grad
from img_utils import to_image
from dataset import create_dataset

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser(description="ClusterGAN Training Script")
parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
parser.add_argument("-i", "--img_size", dest="img_size", type=int, default=28, help="Size of image dimension")
parser.add_argument("-d", "--latent_dim", dest="latent_dim", default=30, type=int, help="Dimension of latent space")
parser.add_argument("-l", "--lr", dest="learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("-c", "--n_critic", dest="n_critic", type=int, default=5, help="Number of training steps for discriminator per iter")
parser.add_argument("-w", "--wass_flag", dest="wass_flag", action='store_true', help="Flag for Wasserstein metric")
args = parser.parse_args()


# Sample a random latent space vector
def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False):
    # assert (fix_class == -1 or (fix_class >= 0 and fix_class < n_c) ), "Requested class %i outside bounds."%fix_class
    
    # Sample noise as generator input, zn
    zn = mnp.randn((shape, latent_dim)) * 0.75

    ######### zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation

    if (fix_class == -1):
        zc_idx = mnp.randint(0, n_c, shape)
        zc_FT = ops.one_hot(zc_idx, n_c, ops.scalar_to_tensor(1.0), ops.scalar_to_tensor(0.0), 1)
    else:
        zc_FT = ops.zeros((shape, n_c), mindspore.float32)
        zc_idx = ops.fill(mindspore.int32, (shape,), fix_class)
        zc_FT[:, fix_class] = 1

    zc = zc_FT

    # Return components of latent space variable
    return zn, zc, zc_idx

def calc_gradient_penalty(real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.shape[0]

    # Calculate interpolation
    alpha = mnp.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    
    interpolated = alpha * real_data + (1 - alpha) * generated_data

    # Calculate gradients of probabilities with respect to examples
    grad_fn = grad(discriminator)
    (gradients,) = grad_fn(interpolated)
    
    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = ops.sqrt(ops.reduce_sum(gradients ** 2, axis=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()

def initialize_weights(top_cell: nn.Cell):
    for _, cell in top_cell.cells_and_names():
        if isinstance(cell, nn.Conv2d) or \
            isinstance(cell, nn.Conv2dTranspose) or \
            isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(0.02), cell.weight.shape))
            cell.bias.set_data(initializer('zeros', cell.bias.shape))

# Softmax function
def softmax(x):
    return ops.Softmax(1)(x)


class Reshape(nn.Cell):
    """
    Class for performing a reshape as a layer in a sequential model.
    """
    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def construct(self, x):
        return x.view(x.shape[0], *self.shape)
    
    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'shape={}'.format(
                self.shape
            )


class Generator_CNN(nn.Cell):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, latent_dim, n_c, x_shape, verbose=False):
        super(Generator_CNN, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = (128, 7, 7)
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose
        
        self.model = nn.SequentialCell(
            # Fully connected layers
            nn.Dense(self.latent_dim + self.n_c, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2),
        
            # Reshape to 128 x (7x7)
            Reshape(self.ishape),

            # Upconvolution layers
            nn.Conv2dTranspose(128, 64, 4, stride=2, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2dTranspose(64, 1, 4, stride=2, padding=1, pad_mode='pad', has_bias=True),
            nn.Sigmoid()
        )

        initialize_weights(self)
        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)
    
    def construct(self, zn, zc):
        z = ops.concat((zn, zc), 1)
        x_gen = self.model(z)
        # Reshape for output
        x_gen = x_gen.view(x_gen.shape[0], *self.x_shape)
        return x_gen


class Encoder_CNN(nn.Cell):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, latent_dim, n_c, verbose=False):
        super(Encoder_CNN, self).__init__()

        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose
        
        self.model = nn.SequentialCell(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, has_bias=True, pad_mode='valid'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, has_bias=True, pad_mode='valid'),
            nn.LeakyReLU(0.2),
            
            # Flatten
            Reshape(self.lshape),
            
            # Fully connected layers
            nn.Dense(self.iels, 1024),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, latent_dim + n_c)
        )

        initialize_weights(self)
        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def construct(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # Separate continuous and one-hot components
        zn = z[:, 0:self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        # Softmax on zc component
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class Discriminator_CNN(nn.Cell):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """            
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()
        
        self.name = 'discriminator'
        self.channels = 1
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        self.verbose = verbose
        
        self.model = nn.SequentialCell(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, has_bias=True, pad_mode='valid'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, has_bias=True, pad_mode='valid'),
            nn.LeakyReLU(0.2),
            
            # Flatten
            Reshape(self.lshape),
            
            # Fully connected layers
            nn.Dense(self.iels, 1024),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, 1),
        )
        
        # If NOT using Wasserstein metric, final Sigmoid
        if (not self.wass):
            self.model = nn.SequentialCell(self.model, nn.Sigmoid())

        initialize_weights(self)
        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def construct(self, img):
        # Get output
        validity = self.model(img)
        return validity


# Training details
n_epochs = args.n_epochs
batch_size = args.batch_size
test_batch_size = 5000
lr = args.learning_rate
b1 = 0.5
b2 = 0.9
decay = 2.5*1e-5
n_skip_iter = args.n_critic

# Data dimensions
img_size = args.img_size
channels = 1

# Latent space info
latent_dim = args.latent_dim
n_c = 10
betan = 10
betac = 10

# Wasserstein+GP metric flag
wass_metric = args.wass_flag

x_shape = (channels, img_size, img_size)

# Loss function
bce_loss = nn.BCELoss(reduction='mean')
xe_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

# Initialize generator and discriminator
generator = Generator_CNN(latent_dim, n_c, x_shape)
encoder = Encoder_CNN(latent_dim, n_c)
discriminator = Discriminator_CNN(wass_metric=wass_metric)
generator.update_parameters_name('generator.')
encoder.update_parameters_name('encoder.')
discriminator.update_parameters_name('discriminator.')

optimizer_GE = nn.Adam(generator.trainable_params() + encoder.trainable_params(),
                       learning_rate=lr, beta1=b1, beta2=b2, weight_decay=decay)
optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=lr, beta1=b1, beta2=b2)
optimizer_GE.update_parameters_name('optim_ge.')
optimizer_D.update_parameters_name('optim_d.')

def generator_forward(zn, zc, zc_idx):
    # Generate a batch of images
    gen_imgs = generator(zn, zc)
    
    # Discriminator output from real and generated samples
    D_gen = discriminator(gen_imgs)
    # Encode the generated images
    enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)

    # Calculate losses for z_n, z_c
    zn_loss = mse_loss(enc_gen_zn, zn)
    zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)

    # Check requested metric
    if wass_metric:
        # Wasserstein GAN loss
        ge_loss = ops.reduce_mean(D_gen) + betan * zn_loss + betac * zc_loss
    else:
        # Vanilla GAN loss
        valid = ops.ones((gen_imgs.shape[0], 1), mindspore.float32)
        v_loss = bce_loss(D_gen, valid)
        ge_loss = v_loss + betan * zn_loss + betac * zc_loss

    return ge_loss

def discriminator_forward(real_imgs, zn, zc):
    # Generate a batch of images
    gen_imgs = generator(zn, zc)
    
    # Discriminator output from real and generated samples
    D_gen = discriminator(gen_imgs)
    D_real = discriminator(real_imgs)

    # Measure discriminator's ability to classify real from generated samples
    if wass_metric:
        # Gradient penalty term
        grad_penalty = calc_gradient_penalty(real_imgs, gen_imgs)

        # Wasserstein GAN loss w/gradient penalty
        d_loss = ops.reduce_mean(D_real) - ops.reduce_mean(D_gen) + grad_penalty
        
    else:
        # Vanilla GAN loss
        valid = ops.ones((gen_imgs.shape[0], 1), mindspore.float32)
        fake = ops.zeros((gen_imgs.shape[0], 1), mindspore.float32)
        real_loss = bce_loss(D_real, valid)
        fake_loss = bce_loss(D_gen, fake)
        d_loss = (real_loss + fake_loss) / 2

    return d_loss


grad_generator_fn = value_and_grad(generator_forward,
                                   optimizer_GE.parameters)
grad_discriminator_fn = value_and_grad(discriminator_forward,
                                       optimizer_D.parameters)

@ms_function
def train_step_d(real_imgs, zn, zc):
    d_loss, d_grads = grad_discriminator_fn(real_imgs, zn, zc)
    optimizer_D(d_grads)
    return d_loss

@ms_function
def train_step_g(zn, zc, zc_idx):
    g_loss, g_grads = grad_generator_fn(zn, zc, zc_idx)
    optimizer_GE(g_grads)

    return g_loss


train_dataset = create_dataset('../../dataset', 'train', args.img_size, args.batch_size)
test_dataset = create_dataset('../../dataset', 'test', args.img_size, args.batch_size)
dataset_size = train_dataset.get_dataset_size()

test_imgs, test_labels = next(test_dataset.create_tuple_iterator())

for epoch in range(args.n_epochs):
    generator.set_train()
    encoder.set_train()
    discriminator.set_train()

    t = tqdm(total=dataset_size)
    t.set_description('Epoch %i' % epoch)
    for i, (imgs, labels) in enumerate(train_dataset.create_tuple_iterator()):
        # Sample random latent variables
        zn, zc, zc_idx = sample_z(shape=imgs.shape[0],
                                  latent_dim=latent_dim,
                                  n_c=n_c)
        # Train the generator every n_critic steps
        if i % n_skip_iter == 0:
            g_loss = train_step_g(zn, zc, zc_idx)
        d_loss = train_step_d(imgs, zn, zc)
        t.set_postfix(g_loss=g_loss, d_loss=d_loss)
        t.update(1)


    # Generator in eval mode
    generator.set_train(False)
    encoder.set_train(False)

    # Set number of examples for cycle calcs
    n_sqrt_samp = 5
    n_samp = n_sqrt_samp * n_sqrt_samp


    ## Cycle through test real -> enc -> gen
    t_imgs, t_label = test_imgs, test_labels
    # Encode sample real instances
    e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
    # Generate sample instances from encoding
    teg_imgs = generator(e_tzn, e_tzc)
    # Calculate cycle reconstruction loss
    img_mse_loss = mse_loss(t_imgs, teg_imgs)

    ## Cycle through randomly sampled encoding -> generator -> encoder
    zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp,
                                             latent_dim=latent_dim,
                                             n_c=n_c)
    # Generate sample instances
    gen_imgs_samp = generator(zn_samp, zc_samp)

    # Encode sample instances
    zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)

    # Calculate cycle latent losses
    lat_mse_loss = mse_loss(zn_e, zn_samp)
    lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)

    # Save cycled and generated examples!
    r_imgs, i_label = imgs[:n_samp], labels[:n_samp]
    e_zn, e_zc, e_zc_logits = encoder(r_imgs)
    reg_imgs = generator(e_zn, e_zc)
    to_image(reg_imgs[:n_samp],
               'images/cycle_reg_%06i.png' %(epoch), 
               nrow=n_sqrt_samp, normalize=True)
    to_image(gen_imgs_samp[:n_samp],
               'images/gen_%06i.png' %(epoch), 
               nrow=n_sqrt_samp, normalize=True)
    
    ## Generate samples for specified classes
    stack_imgs = []
    for idx in range(n_c):
        # Sample specific class
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_c,
                                                 latent_dim=latent_dim,
                                                 n_c=n_c,
                                                 fix_class=idx)

        # Generate sample instances
        gen_imgs_samp = generator(zn_samp, zc_samp)

        if (len(stack_imgs) == 0):
            stack_imgs = gen_imgs_samp
        else:
            stack_imgs = ops.concat((stack_imgs, gen_imgs_samp), 0)

    # Save class-specified generated examples!
    to_image(stack_imgs,
               'images/gen_classes_%06i.png' %(epoch),
               nrow=n_c, normalize=True)
    
    print("\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]"%(img_mse_loss.asnumpy(), 
                                                         lat_mse_loss.asnumpy(), 
                                                         lat_xe_loss.asnumpy())
         )

# MindSpore-GAN
<p align="center"><img width="480" src="https://raw.githubusercontent.com/mindspore-ai/mindspore/master/docs/MindSpore-logo.png" /></p>

Collection of MindSpore implementations of Generative Adversarial Network varieties presented in research papers. Models are refered to [Pytorch-GAN](https://github.com/eriklindernoren/Pytorch-GAN).

## Table of Contents
  * [Installation](#installation)
  * [Datasets](#datasets)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#auxiliary-classifier-gan)
    + [Adversarial Autoencoder](#adversarial-autoencoder)
    + [BEGAN](#began)
    <!-- + [BicycleGAN](#bicyclegan) -->
    + [Boundary-Seeking GAN](#boundary-seeking-gan)
    + [Cluster GAN](#cluster-gan) 
    + [Conditional GAN](#conditional-gan)
    <!-- + [Context-Conditional GAN](#context-conditional-gan)
    + [Context Encoder](#context-encoder)
    + [Coupled GAN](#coupled-gan)
    + [CycleGAN](#cyclegan)
    + [Deep Convolutional GAN](#deep-convolutional-gan)
    + [DiscoGAN](#discogan)
    + [DRAGAN](#dragan)
    + [DualGAN](#dualgan)
    + [Energy-Based GAN](#energy-based-gan)
    + [Enhanced Super-Resolution GAN](#enhanced-super-resolution-gan) -->
    + [GAN](#gan)
    <!-- + [InfoGAN](#infogan)
    + [Least Squares GAN](#least-squares-gan)
    + [MUNIT](#munit)
    + [Pix2Pix](#pix2pix)
    + [PixelDA](#pixelda)
    + [Relativistic GAN](#relativistic-gan)
    + [Semi-Supervised GAN](#semi-supervised-gan)
    + [Softmax GAN](#softmax-gan)
    + [StarGAN](#stargan)
    + [Super-Resolution GAN](#super-resolution-gan)
    + [UNIT](#unit)
    + [Wasserstein GAN](#wasserstein-gan) -->
    + [Wasserstein GAN GP](#wasserstein-gan-gp)
    <!-- + [Wasserstein GAN DIV](#wasserstein-gan-div) -->

## Installation

```
    $ git clone https://github.com/lvyufeng/MindSpore-GAN
    $ cd MindSpore-GAN/
    $ pip3 install -r requirements.txt
```

## Datasets

Download all datasets for GAN before you try to run the examples.

```bash
$ bash scripts/download_mnist.sh
```

## Implementations

### Auxiliary Classifier GAN
_Auxiliary Classifier Generative Adversarial Network_

#### Authors
Augustus Odena, Christopher Olah, Jonathon Shlens

#### Abstract
Synthesizing high resolution photorealistic images has been a long-standing challenge in machine learning. In this paper we introduce new methods for the improved training of generative adversarial networks (GANs) for image synthesis. We construct a variant of GANs employing label conditioning that results in 128x128 resolution image samples exhibiting global coherence. We expand on previous work for image quality assessment to provide two new analyses for assessing the discriminability and diversity of samples from class-conditional image synthesis models. These analyses demonstrate that high resolution samples provide class information not present in low resolution samples. Across 1000 ImageNet classes, 128x128 samples are more than twice as discriminable as artificially resized 32x32 samples. In addition, 84.7% of the classes have samples exhibiting diversity comparable to real ImageNet data.

[[Paper]](https://arxiv.org/abs/1610.09585) [[Code]](src/acgan/acgan.py)

#### Run Example
```
$ cd src/acgan/
$ python3 acgan.py
```

<p align="center">
    <img src="assets/acgan.gif" width="360"\>
</p>


### Adversarial Autoencoder
_Adversarial Autoencoder_

#### Authors
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey

#### Abstract
n this paper, we propose the "adversarial autoencoder" (AAE), which is a probabilistic autoencoder that uses the recently proposed generative adversarial networks (GAN) to perform variational inference by matching the aggregated posterior of the hidden code vector of the autoencoder with an arbitrary prior distribution. Matching the aggregated posterior to the prior ensures that generating from any part of prior space results in meaningful samples. As a result, the decoder of the adversarial autoencoder learns a deep generative model that maps the imposed prior to the data distribution. We show how the adversarial autoencoder can be used in applications such as semi-supervised classification, disentangling style and content of images, unsupervised clustering, dimensionality reduction and data visualization. We performed experiments on MNIST, Street View House Numbers and Toronto Face datasets and show that adversarial autoencoders achieve competitive results in generative modeling and semi-supervised classification tasks.

[[Paper]](https://arxiv.org/abs/1511.05644) [[Code]](src/aae/aae.py)

#### Run Example
```
$ cd src/aae/
$ python3 aae.py
```

### BEGAN
_BEGAN: Boundary Equilibrium Generative Adversarial Networks_

#### Authors
David Berthelot, Thomas Schumm, Luke Metz

#### Abstract
We propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based Generative Adversarial Networks. This method balances the generator and discriminator during training. Additionally, it provides a new approximate convergence measure, fast and stable training and high visual quality. We also derive a way of controlling the trade-off between image diversity and visual quality. We focus on the image generation task, setting a new milestone in visual quality, even at higher resolutions. This is achieved while using a relatively simple model architecture and a standard training procedure.

[[Paper]](https://arxiv.org/abs/1703.10717) [[Code]](src/began/began.py)

#### Run Example
```
$ cd src/began/
$ python3 began.py
```

### Boundary-Seeking GAN
_Boundary-Seeking Generative Adversarial Networks_

#### Authors
R Devon Hjelm, Athul Paul Jacob, Tong Che, Adam Trischler, Kyunghyun Cho, Yoshua Bengio

#### Abstract
Generative adversarial networks (GANs) are a learning framework that rely on training a discriminator to estimate a measure of difference between a target and generated distributions. GANs, as normally formulated, rely on the generated samples being completely differentiable w.r.t. the generative parameters, and thus do not work for discrete data. We introduce a method for training GANs with discrete data that uses the estimated difference measure from the discriminator to compute importance weights for generated samples, thus providing a policy gradient for training the generator. The importance weights have a strong connection to the decision boundary of the discriminator, and we call our method boundary-seeking GANs (BGANs). We demonstrate the effectiveness of the proposed algorithm with discrete image and character-based natural language generation. In addition, the boundary-seeking objective extends to continuous data, which can be used to improve stability of training, and we demonstrate this on Celeba, Large-scale Scene Understanding (LSUN) bedrooms, and Imagenet without conditioning.

[[Paper]](https://arxiv.org/abs/1702.08431) [[Code]](src/bgan/bgan.py)

#### Run Example
```
$ cd src/bgan/
$ python3 bgan.py
```

### Cluster GAN
_ClusterGAN: Latent Space Clustering in Generative Adversarial Networks_

#### Authors
Sudipto Mukherjee, Himanshu Asnani, Eugene Lin, Sreeram Kannan

#### Abstract
Generative Adversarial networks (GANs) have obtained remarkable success in many unsupervised learning tasks and
unarguably, clustering is an important unsupervised learning problem. While one can potentially exploit the
latent-space back-projection in GANs to cluster, we demonstrate that the cluster structure is not retained in the
GAN latent space.  In this paper, we propose ClusterGAN as a new mechanism for clustering using GANs. By sampling
latent variables from a mixture of one-hot encoded variables and continuous latent variables, coupled with an
inverse network (which projects the data to the latent space) trained jointly with a clustering specific loss, we
are able to achieve clustering in the latent space. Our results show a remarkable phenomenon that GANs can preserve
latent space interpolation across categories, even though the discriminator is never exposed to such vectors. We
compare our results with various clustering baselines and demonstrate superior performance on both synthetic and
real datasets.

[[Paper]](https://arxiv.org/abs/1809.03627) [[Code]](src/clustergan/clustergan.py)

Code based on a full PyTorch [[implementation]](https://github.com/zhampel/clusterGAN).

#### Run Example
```
$ cd src/clustergan/
$ python3 clustergan.py
```

> Notice: The ClusterGAN model can not train currectly with same implementation from Pytorch. All modules include Discriminator, Generator and Encoder are tested with same initialized parameters. It seems that 'do `loss.backward` twice' can not be simplified to two gradient function.

### Conditional GAN
_Conditional Generative Adversarial Nets_

#### Authors
Mehdi Mirza, Simon Osindero

#### Abstract
Generative Adversarial Nets [8] were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.

[[Paper]](https://arxiv.org/abs/1411.1784) [[Code]](src/cgan/cgan.py)

#### Run Example
```
$ cd src/cgan/
$ python3 cgan.py
```

### GAN
_Generative Adversarial Network_

#### Authors
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

#### Abstract
We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

[[Paper]](https://arxiv.org/abs/1406.2661) [[Code]](src/gan/gan.py)

#### Run Example
```bash
$ cd src/gan/
$ python3 gan.py
```

### Wasserstein GAN GP
_Improved Training of Wasserstein GANs_

#### Authors
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville

#### Abstract
Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only low-quality samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models over discrete data. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

[[Paper]](https://arxiv.org/abs/1704.00028) [[Code]](src/wgan_gp/wgan_gp.py)

#### Run Example
```bash
$ bash scripts/download_mnist.sh
$ cd src/wgan_gp/
$ python3 wgan_gp.py
```
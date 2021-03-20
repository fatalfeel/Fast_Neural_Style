# fast-neural-style: Fast Style Transfer in Pytorch! :art:

An implementation of **fast-neural-style** in PyTorch! Style Transfer learns the aesthetic style of a `style image`, usually an art work, and applies it on another `content image`. This repository contains codes the can be used for:
1. fast `image-to-image` aesthetic style transfer, 
2. `image-to-video` aesthetic style transfer, and for
3. training `style-learning` transformation network

This implemention follows the style transfer approach outlined in [**Perceptual Losses for Real-Time Style Transfer and Super-Resolution**](https://arxiv.org/abs/1603.08155) paper by *Justin Johnson, Alexandre Alahi, and Fei-Fei Li*, along with the [supplementary paper detailing the exact model architecture](https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf) of the mentioned paper. The idea is to train a **`separate feed-forward neural network (called Transformation Network) to transform/stylize`** an image and use backpropagation to learn its parameters, instead of directly manipulating the pixels of the generated image as discussed in [A Neural Algorithm of Artistic Style aka **neural-style**](https://arxiv.org/abs/1508.06576) paper by *Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge*. The use of feed-forward transformation network allows for fast stylization of images, around 1000x faster than neural style.
# Prepare data
./install_data.sh

# Comparison of Transformer Networks on experimental.py

|                       Network                      | size (Kb) | no. of parameters | final loss (million) |
|:---------------------------------------------------|----------:|------------------:|---------------------:|
| transformer/TransformerNetwork                     |     6,573 |         1,679,235 |                 9.88 |
| experimental/TransformerNetworkDenseNet            |     1,064 |           269,731 |                11.37 |
| experimental/TransformerNetworkUNetDenseNetResNet  |     1,062 |           269,536 |                12.32 |
| experimental/TransformerNetworkV2                  |     6,573 |         1,679,235 |                10.05 |
| experimental/TransformerResNextNetwork             |     1,857 |           470,915 |                10.31 |
| experimental/TransformerResNextNetwork_Pruned(0.3) |        44 |             8,229 |                19.29 |
| experimental/TransformerResNextNetwork_Pruned(1.0) |       260 |            63,459 |                12.72 |


`TransformerResNextNetwork` and `TransformerResNextNetwork_Pruned(1.0)` provides the best tradeoff between compute, memory size, and performance.

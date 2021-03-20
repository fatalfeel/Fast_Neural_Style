# fast-neural-style: Fast Style Transfer in Pytorch!

An implementation of **fast-neural-style** in PyTorch! Style Transfer learns the aesthetic style of a `style image`, usually an art work, and applies it on another `content image`. This repository contains codes the can be used for:
1. fast `image-to-image` aesthetic style transfer, 
2. `image-to-video` aesthetic style transfer, and for
3. training `style-learning` transformation network

# Prepare data
- ./install_data.sh
- or
- bash -x ./install_data.sh

# Train data
python3 ./train.py --cuda True

# Test data
- python3 ./stylize.py --cuda True
- python3 ./video.py --cuda True

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

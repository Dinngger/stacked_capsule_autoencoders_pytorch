# Stacked Capsule Autoencoders PyTorch
An unofficial implementation of the paper ["Stacked Capsule Autoencoders"](https://arxiv.org/abs/1906.06818) in PyTorch.

  * **Author**: Dinger, College of Artificial Intelligence, Xi'an Jiaotong University
  * **Email**: dinger@stu.xjtu.edu.cn

This repository aims to reproducing the original paper in pytorch and to be closed to the original tensorflow implementation as much as possible.

## Dependencies

This repository is based on pytorch 1.8 but sure you can use an older version.

    pip install -r requirements.txt

## Running Experiments

You can train the model by

    python train.py

and use tensorboard to see the loss and accuracy.

    tensorboard --logdir ./checkpoints

## Existing Issues

- The accuracy can only reach about 40%, while the original implementation can reach 97% accuracy.
- Cuda memory will increase a little after each epoch. I haven't found the reason yet.

# pytorch_usage-distributed
An example of using PyTorch [Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html) (on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) classification)

+ 1 node + N GPUs
```
python -m torch.distributed.launch --nproc_per_node=N main.py
```

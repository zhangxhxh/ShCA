# Semi-hard constraint augmentation of triplet learning to improve image corruption classification


## Dataset
You can download the datasets from the links below:

+ [cifar10](https://paperswithcode.com/dataset/cifar-10).
+ [cifar100](https://paperswithcode.com/dataset/cifar-100).
+ [cifar100-C and cifar10-C](https://zenodo.org/records/3555552).
  
please download these datasets to ./data/CIFAR-10-C and ./data/CIFAR-100-C.

## Environments
Currently, requires following packages
- python 3.8
- CUDA Version: 11.7
- PyTorch 1.11.0
- torchvision 0.10.0
- scikit-learn 1.0.1

## Training & Evaluation
To train the models in paper, run this command:
```train
python main.py
```
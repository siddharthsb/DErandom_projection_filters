# Adversarial Robustness via Random Projection Filters

## Environment

* torch 1.7.1
* torchvision 0.8.2
* torchattacks 3.2.6

## Training of RPF

* To train a ResNet18 with RPF on CIFAR-10 with fixed random projection filteer:
```
python train.py --network ResNet18 --dataset cifar10 --attack_iters 10 --lr_schedule multistep --epochs 201 --adv_training --rp --rp_block -1 -1 --rp_out_channel 48 --rp_weight_decay 1e-2 --save_dir resnet18_c10_RPF --model_num 0
```

* To train models with changing RPF's refer to Dong and Xu, github(https://github.com/uniserj/random-projection-filters)

## Evaluation of RPF

* To evaluate the performance of multiple ResNet18 with fixed RPF on CIFAR-10:

```
python evaluate_multiple.py --dataset cifar10 --network ResNet18 --rp --rp_out_channel 48 --rp_block -1 -1 --save_dir eval_r18_c10 --pretrain resnet18_c10_RPF/ --num_models 14 --start_from 0
```

Models in pretrain directory must be named weight_%n_latest.pth wherre %n is the model number.

* To evaluate the performance of a ResNet 18 with changing RPF on CIFAR-10:

```
python evaluate.py --dataset cifar10 --network ResNet18 --rp --rp_out_channel 48 --rp_block -1 -1 --save_dir eval_r18_c10 --pretrain resnet18_c10_RPF/resnet18_c10.pth --num_trials 30
```

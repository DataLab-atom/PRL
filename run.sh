# nohup python test_all_cifar.py -r saved/cifar100/ir100/bs_e200_tau1.0_bs128_lr0.1/models/Imbalance_CIFAR100LT_IR100/0310_203016/checkpoint-epoch200.pth -c configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json > result3.log 2>&1


# python test_all_cifar_new.py -r saved/cifar100/ir100/bs_e200_tau1.0_bs128_lr0.1/models/Imbalance_CIFAR100LT_IR100/0310_203016/checkpoint-epoch200.pth -c configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json --posthoc_bias_correction



# nohup python test_all_cifar_new.py -r saved/cifar100/ir100/bs_e200_tau1.0_bs128_lr0.1/models/Imbalance_CIFAR100LT_IR100/0310_203016/checkpoint-epoch200.pth -c configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json --posthoc_bias_correction  > result6.log 2>&1

# nohup python t_get_ray.py -r saved/cifar100/ir100/bs_e200_tau1.0_bs128_lr0.1/models/Imbalance_CIFAR100LT_IR100/0310_203016/checkpoint-epoch200.pth -c configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json --posthoc_bias_correction  > result7.log 2>&1 &



# nohup python t_get_ray.py -r saved/cifar100/ir100/bs_e200_tau1.0_bs128_lr0.1/models/Imbalance_CIFAR100LT_IR100/0312_151351/checkpoint-epoch200.pth -c configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json --posthoc_bias_correction > result_top_100.log 2>&1 &

# nohup python train.py ---c ./configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json > result.log 2>&1 &

# nohup python test_training_cifar.py -c ./configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json -r saved/cifar100/ir100/bs_e200_tau1.0_bs128_lr0.1/models/Imbalance_CIFAR100LT_IR100/0809_030949/checkpoint-epoch200.pth > result_new.log 2>&1 &


# nohup python test_training_cifar.py -c ./configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json -r saved/cifar100/ir100/bs_e200_tau1.0_bs128_lr0.1/models/Imbalance_CIFAR100LT_IR100/0312_151351/checkpoint-epoch200.pth > result_new3.log 2>&1 &


# nohup python test_all_cifar.py -c ./configs/mixup/standard_training/config_cifar100_ir100_bs-experts.json -r saved/cifar100/ir100/bs_e200_tau1.0_bs128_lr0.1/models/Imbalance_CIFAR100LT_IR100/0312_151351/checkpoint-epoch200.pth > result_new4.log 2>&1 &

# nohup python test.py -r saved/cifar100/ir100/bs_e200_tau1.0_bs128_lr0.1/models/Imbalance_CIFAR100LT_IR100/0312_151351/checkpoint-epoch200.pth > result_new5.log 2>&1 &
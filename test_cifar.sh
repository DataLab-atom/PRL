# CUDA_VISIBLE_DEVICES=5 python train.py -c configs/config_cifar100_ir100_sade_sum.json > logs/log_tring_mu0.1_a0.5.log 2>&1
# CUDA_VISIBLE_DEVICES=5 python train.py -c configs/config_cifar100_ir100_sade_hnet_sum.json > logs/log_tring_hent_sum.log 2>&1
# CUDA_VISIBLE_DEVICES=5 python train.py -c configs/config_cifar100_ir100_sade_hnet_stch.json > logs/test_log_tring_hent_stch_mu0.10_a1.2.log 2>&1

CUDA_VISIBLE_DEVICES="2" python train.py -c configs/config_imagenet_lt_resnext50_sade.json > logs/log_imagenet_SADE.log 2>&1
CUDA_VISIBLE_DEVICES="2" python train.py -c configs/config_iNaturalist_resnet50_sade.json > logs/log_inat_SADE.log 2>&1
# CUDA_VISIBLE_DEVICES="1,4,5,6" python train.py -c configs/config_imagenet_lt_resnext50_hnet.json > logs/log_imagenet_hnet.log 2>&1


import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
from data_loader.imagenet_lt_data_loaders import ImageNetLTDataLoader
from data_loader.cifar_data_loaders import TestAgnosticImbalanceCIFAR100DataLoader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.nn.functional as F

def main(config,args):
    
    ray = torch.from_numpy(np.array(args.ray.split(',')).astype(float)).float().cuda()
    print(ray)
    logger = config.get_logger('test')

    # setup data_loader instances 
    if 'returns_feat' in config['arch']['args']:
        model = config.init_obj('arch', module_arch, allow_override=True, returns_feat=False,use_hnet=config['use_hnet'])
    else:
        model = config.init_obj('arch', module_arch)
    #logger.info(model)
 
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
  
    train_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        training=True,
        num_workers=8,
        imb_factor=config['data_loader']['args']['imb_factor']
    )    
    train_cls_num_list = train_data_loader.cls_num_list 
    train_cls_num_list=torch.tensor(train_cls_num_list)
    many_shot = train_cls_num_list > 100
    few_shot = train_cls_num_list <20
    medium_shot =~many_shot & ~few_shot
    
    num_classes = config._config["arch"]["args"]["num_classes"]
    
    distrb = {
        'uniform': (0,False),
        'forward50': (0.02, False),
        'forward25': (0.04, False), 
        'backward50': (0.02, True),
        'backward25': (0.04, True),
    }  
    
    record_list,save_acc=[],[]
    
    test_distribution_set = ["forward50",  "forward25", "uniform", "backward25", "backward50"] 
    for test_distribution in test_distribution_set: 
        print(test_distribution)
        data_loader = TestAgnosticImbalanceCIFAR100DataLoader(
            config['data_loader']['args']['data_dir'],
            batch_size=128,
            shuffle=False,
            training=False,
            num_workers=2,
            test_imb_factor=distrb[test_distribution][0],
            reverse=distrb[test_distribution][1]
        )
        record = validation(data_loader, model, num_classes,device, many_shot, medium_shot, few_shot,ray = ray,save_acc=save_acc)
            
        record_list.append(record)
    print('='*25, ' Final results ', '='*25)
    i = 0
    for txt in record_list:
        print(test_distribution_set[i]+'\t')
        print(*txt)          
        i+=1
    
    np.save('grid_logs/ray_{}_.npy'.format(args.ray),np.array(save_acc))

def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                       + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1
   

def validation(data_loader, model, num_classes,device,many_shot, medium_shot, few_shot,ray,save_acc):
 
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    total_logits = torch.empty((0, num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data,ray)
            for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target))  
             
    probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)

    # Calculate the overall accuracy and F measurement
    eval_acc_mic_top1= mic_acc_cal(preds[total_labels != -1],
                                        total_labels[total_labels != -1])
         
    acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)

    acc = acc_per_class.cpu().numpy() 
    save_acc.append(acc)
    many_shot_acc = acc[many_shot].mean()
    medium_shot_acc = acc[medium_shot].mean()
    few_shot_acc = acc[few_shot].mean()
    print("Many-shot {}, Medium-shot {}, Few-shot {}, All {}".format(np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2), np.round(eval_acc_mic_top1 * 100, decimals=2)))
    return np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2), np.round(eval_acc_mic_top1 * 100, decimals=2)
 

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='configs/test_time_cifar100_ir100_sade_hnet.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='saved/cifar100/ir100/sade_e200_inv2_bs128_lr0.1/models/Imbalance_CIFAR100LT_IR100_SADE/0520_030244/model_best.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='7', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--ray', default='0,0,1', type=str)

    config,args = ConfigParser.from_args(args,reargs=True)
    main(config,args)

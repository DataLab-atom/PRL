import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
from data_loader.cifar_data_loaders import TestAgnosticImbalanceCIFAR100DataLoader
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.nn.functional as F
from utils import adjusted_model_wrapper

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import copy
from sentence_transformers import SentenceTransformer, util
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(config, posthoc_bias_correction=False):
    logger = config.get_logger('test')

    # setup data_loader instances 
    if 'returns_feat' in config['arch']['args']:
        model = config.init_obj('arch', module_arch, allow_override=True, returns_feat=False, use_hnet=config['use_hnet'])
    else:
        model = config.init_obj('arch', module_arch, use_hnet=config['use_hnet'])
 
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
    few_shot =train_cls_num_list <20
    medium_shot =~many_shot & ~few_shot 

    num_classes = config._config["arch"]["args"]["num_classes"]
    
    distrb = {
        'uniform': (0,False),
        'forward50': (0.02, False),
        'forward25': (0.04, False), 
        'forward10':(0.1, False),
        'forward5': (0.2, False),
        'forward2': (0.5, False),
        'backward50': (0.02, True),
        'backward25': (0.04, True),
        'backward10': (0.1, True),
        'backward5': (0.2, True),
        'backward2': (0.5, True),
    }  
    
    record_list=[]
    
    test_distribution_set = ["forward50",  "forward25", "forward10", "forward5", "forward2", "uniform",  "backward2", "backward5", "backward10", "backward25", "backward50"] 
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

        if posthoc_bias_correction:
            test_prior = torch.tensor(data_loader.cls_num_list).float().to(device)
            test_prior = test_prior / test_prior.sum()
            test_bias = test_prior.log()
        else:
            test_bias = None

        adjusted_model = adjusted_model_wrapper(model, test_bias=test_bias)

        weight = [config['aggregation_weight1'], config['aggregation_weight2'], config['aggregation_weight3']]
        
        record = validation(data_loader, adjusted_model, num_classes,device, many_shot, medium_shot, few_shot, weight)
            
        record_list.append(record)
    print('='*25, ' Final results ', '='*25)
    i = 0
    for txt in record_list:
        print(test_distribution_set[i]+'\t')
        print(*txt)          
        i+=1
    
    # 提取所有分布的 eval_acc_mic_top1（即返回结果的最后一个值）
    eval_acc_mic_top1_list = [record[3] for record in record_list]
    # 计算平均值
    average_eval_acc_mic_top1 = np.mean(eval_acc_mic_top1_list)
    # print(average_eval_acc_mic_top1)
    return average_eval_acc_mic_top1, record_list    

def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                       + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1
   

def validation(data_loader, model, num_classes,device,many_shot, medium_shot, few_shot, weight):
 
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    total_logits = torch.empty((0, num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            weight = torch.tensor(weight).cuda()
            output = model(data, ray=weight)
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
    many_shot_acc = acc[many_shot].mean()
    medium_shot_acc = acc[medium_shot].mean()
    few_shot_acc = acc[few_shot].mean()
    print("Many-shot {}, Medium-shot {}, Few-shot {}, All {}".format(np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2), np.round(eval_acc_mic_top1 * 100, decimals=2)))
    return np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2), np.round(eval_acc_mic_top1 * 100, decimals=2)
 

# 定义一个简单的全连接神经网络 (DNN)
class DNN(nn.Module):
    def __init__(self, input_dim=384, output_dim=3):
        super(DNN, self).__init__()
        
        # 定义网络结构
        self.fc1 = nn.Linear(input_dim, 512)  # 第一层：输入 384 维，输出 512 维
        self.fc2 = nn.Linear(512, 256)        # 第二层：输入 512 维，输出 256 维
        self.fc3 = nn.Linear(256, 128)        # 第三层：输入 256 维，输出 128 维
        self.fc4 = nn.Linear(128, output_dim) # 第四层：输入 128 维，输出 3 维
        
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 定义前向传播过程
        x = self.relu(self.fc1(x))  # 第一层 + ReLU
        x = self.relu(self.fc2(x))  # 第二层 + ReLU
        x = self.relu(self.fc3(x))  # 第三层 + ReLU
        x = self.fc4(x)             # 第四层输出
        return x


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-l', '--log-config', default='logger/logger_config.json', type=str,
                      help='logging config file path (default: logger/logger_config.json)')
    args.add_argument("--posthoc_bias_correction", dest="posthoc_bias_correction", action="store_true", default=False)

    # dummy arguments used during training time
    args.add_argument("--validate")
    args.add_argument("--use-wandb")

    config, args = ConfigParser.from_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建一个模型实例
    model = DNN(input_dim=384, output_dim=3)
    model.load_state_dict(torch.load("dnn_text_to_vector.pth"))
    model.to(device)
    model.eval()  # 切换到评估模式

    test_text = "Uniform distribution (balanced classes) variant 1"
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    test_text = semantic_model.encode(test_text,  convert_to_tensor=True)    # 文本嵌入
    
    with torch.no_grad():

        predicted_vector = model(test_text.to(device))
    # print("Input text:", test_text)
    print("Predicted vector:", predicted_vector.cpu().numpy())
    predicted_vector = predicted_vector.cpu().numpy()
    #  加一个归一化
    print(predicted_vector.shape)
    w1 = predicted_vector[0]
    w2 = predicted_vector[1]
    w3 = predicted_vector[2]
    config_copy = copy.deepcopy(config)
    # config._config['aggregation_weight1'], ['aggregation_weight2'], ['aggregation_weight3']
    config_copy._config['aggregation_weight1'] = w1
    config_copy._config['aggregation_weight2'] = w2
    config_copy._config['aggregation_weight3'] = w3

    # 调用测试函数，获得当前组合下的平均评测准确率
    score, record_list = main(config_copy, args.posthoc_bias_correction)

    print("="*30, "最终结果", "="*30)
    print(f"测试分布对应的偏好向量: w1={w1}, w2={w2}, w3={w3}  得分: {score:.4f}")
    print("详细性能记录:")
    for rec in record_list:
        print(rec)
    print("-" * 60)
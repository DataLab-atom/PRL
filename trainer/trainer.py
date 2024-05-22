import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from test_training_cifar import TestAgnosticImbalanceCIFAR100DataLoader
from utils import inf_loop, MetricTracker, load_state_dict, rename_parallel_state_dict, autocast, use_fp16
import model.model as module_arch
from phn import get_Solver
import torch.nn.functional as F
from tqdm import tqdm

from thop import profile
from thop import clever_format

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, alpha=1):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config

        # add_extra_info will return info about individual experts. This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = config._config.get('add_extra_info', False)
        print("self.data_loader",data_loader)

        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        if use_fp16:
            self.logger.warn("FP16 is enabled. This option should be used with caution unless you make sure it's working and we do not provide guarantee.")
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.use_hnet = config["use_hnet"]
        self.alpha = alpha

        solvers_config = config['Solvers']
        solvers_config['n_params'] = count_parameters(self.model.backbone)
        self.solver = get_Solver(solvers_config)
        try:
            self.solver_share_model = self.model.module.backbone
        except:
            self.solver_share_model = self.model.backbone

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        if len(self.device_ids) > 1:
            print(self.device_ids)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.real_model._hook_before_iter()
        self.train_metrics.reset()

        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)

        for batch_idx, data in enumerate(self.data_loader):

            data, target = data
            data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()

            with autocast():
                if self.real_model.requires_target:
                    output = self.model(data, target=target) 
                    output, loss = output   
                else:
                    # print("this way....")
                    extra_info = {}
                    output = self.model(data)

                    if self.add_extra_info:
                        if isinstance(output, dict):
                            logits = output["logits"]
                            extra_info.update({
                                "logits": logits.transpose(0, 1)
                            })
                        else:
                            extra_info.update({
                                "logits": self.real_model.backbone.logits
                            })

                    if isinstance(output, dict):
                        output = output["output"]

                    if self.add_extra_info:
                        loss = self.criterion(output_logits=output, target=target, extra_info=extra_info, use_hnet=self.use_hnet)
                    else:
                        loss = self.criterion(output_logits=output, target=target, use_hnet=self.use_hnet) 

                    if self.use_hnet:
                        losses = torch.stack(loss)
                        # print(losses.shape)
                        loss = self.solver(losses, 0.0, list(self.solver_share_model.parameters()))
    
            if not use_fp16:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, return_length=True))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} max group LR: {:.4f} min group LR: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    max([param_group['lr'] for param_group in self.optimizer.param_groups]),
                    min([param_group['lr'] for param_group in self.optimizer.param_groups])))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        self.new_test()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    ################ begin add 
    def new_test(self):
        # ray = torch.from_numpy(np.array(self.config['ray'].split(',')).astype(float)).float().cuda()
        # print(ray)
        ray = torch.tensor([0.2500, 0.1200, 0.6200])
        ray = ray.cuda()
        train_cls_num_list = self.data_loader.cls_num_list
        #b = np.load("../data/shot_list.npy")
        train_cls_num_list=torch.tensor(train_cls_num_list)
        many_shot = train_cls_num_list > 100
        few_shot =train_cls_num_list <20
        medium_shot =~many_shot & ~few_shot
        num_classes = self.config._config["arch"]["args"]["num_classes"]
        
        distrb = {
            'uniform': (1,False),
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

        # prepare model for testing 
        weight_record_list=[]
        performance_record_list=[]
        record_list,save_acc=[],[]
        test_distribution_set = ["forward50",  "forward25", "forward10", "forward5", "forward2", "uniform",  "backward2", "backward5", "backward10", "backward25", "backward50"] 
        for test_distribution in test_distribution_set:    
            print(test_distribution)
            data_loader = TestAgnosticImbalanceCIFAR100DataLoader(
                self.config['data_loader']['args']['data_dir'],
                batch_size=128,
                shuffle=False,
                training=False,
                num_workers=2,
                test_imb_factor=distrb[test_distribution][0],
                reverse=distrb[test_distribution][1]
            )
            
            valid_data_loader = data_loader.test_set()
            num_classes = self.config._config["arch"]["args"]["num_classes"]        

            record = self.test_validation(valid_data_loader, num_classes, many_shot, medium_shot, few_shot, ray,save_acc=save_acc)    
            performance_record_list.append(record)
        
        print('\n')        
        print('='*25, ' Final results ', '='*25)
        print('\n')
        i = 0
        print('Top-1 accuracy on many-shot, medium-shot, few-shot and all classes:')
        for txt in performance_record_list:
            print(test_distribution_set[i]+'\t')
            print(*txt)          
            i+=1        
        self.model.train()
            
    def test_validation(self, data_loader, num_classes, many_shot, medium_shot, few_shot, ray, save_acc):
        self.model.eval()  
        confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
        total_logits = torch.empty((0, num_classes)).cuda()
        total_labels = torch.empty(0, dtype=torch.long).cuda()
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.cuda(), target.cuda()
                output = self.model(data,ray)['output']
                for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                total_logits = torch.cat((total_logits, output))
                total_labels = torch.cat((total_labels, target))  
                
        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)

        # Calculate the overall accuracy and F measurement
        print(preds.shape)
        print(total_labels.shape)
        eval_acc_mic_top1= self.mic_acc_cal(preds[total_labels != -1], total_labels[total_labels != -1])
            
        acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)

        acc = acc_per_class.cpu().numpy() 
        save_acc.append(acc)
        many_shot_acc = acc[many_shot].mean()
        medium_shot_acc = acc[medium_shot].mean()
        few_shot_acc = acc[few_shot].mean()
        print("Many-shot {}, Medium-shot {}, Few-shot {}, All {}".format(np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2), np.round(eval_acc_mic_top1 * 100, decimals=2)))
        return np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2), np.round(eval_acc_mic_top1 * 100, decimals=2)
    
    def mic_acc_cal(self, preds, labels):
        if isinstance(labels, tuple):
            assert len(labels) == 3
            targets_a, targets_b, lam = labels
            acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                        + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
        else:
            acc_mic_top1 = (preds == labels).sum().item() / len(labels)
        return acc_mic_top1

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
           
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.cuda(), target.cuda()

                if self.real_model.requires_target:
                    output = self.model(data, target=target) 
                    output, loss = output   
                else:
                    # print("this way....")
                    extra_info = {}
                    output = self.model(data)
                    if self.add_extra_info:
                        if isinstance(output, dict):
                            logits = output["logits"]
                            extra_info.update({"logits": logits.transpose(0, 1)})
                        else:
                            extra_info.update({"logits": self.real_model.backbone.logits})

                if isinstance(output, dict):
                    output = output["output"]
                # loss = self.criterion(output, target)
                if self.add_extra_info:
                    loss = self.criterion(output_logits=output, target=target, extra_info=extra_info, use_hnet=self.use_hnet)
                else:
                    loss = self.criterion(output_logits=output, target=target, use_hnet=self.use_hnet) 

                losses = 0
                if self.use_hnet:
                    losses = sum(loss)
                else:
                    losses = loss
                # print(losses)  

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', losses.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

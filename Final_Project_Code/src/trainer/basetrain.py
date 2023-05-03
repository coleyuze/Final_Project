import logging
import os
import time
import torch
from tqdm import tqdm
from src.config import parse_args
from src.dataset import FinetuneDataset
# from src.
# from src.models import MultiModal
from src.utils import *
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR

import warnings
warnings.filterwarnings("ignore")

class BaseTrainer:
    def __init__(self, args) -> None:
        self.args = args
        setup_logging()
        setup_device(args)
        setup_seed(args)
        self.SetEverything(args)
        
    def SetEverything(self,args):
        self.get_model()
        self.get_dataloader()
        if args.embedding_freeze:
            freeze(self.model.roberta.embeddings)
            print("Frozen word embedding parameters，not trained!")
        args.max_steps = args.max_epochs * len(self.train_dataloader)
        args.warmup_steps = int(args.warmup_rate * args.max_steps)
        self.optimizer, self.scheduler = build_optimizer(args, self.model)
        self.model.to(args.device)
        if self.args.swa_start > 0:
            print("use SWA！")
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, args.swa_lr)
            self.swa_model.to(args.device)
        if self.args.ema_start >= 0:
            print("use EMA！")
            self.ema = EMA(self.model, args.ema_decay)
            self.ema.register()
        
        self.resume()
        if args.device == 'cuda':
            if args.distributed_train:
                print("Multi-gpu training!")
                self.model = torch.nn.parallel.DataParallel(self.model)
        if self.args.fgm != 0:
            logging.info("FGM is used for embedding attack！")
            self.fgm = FGM(self.model.module.roberta.embeddings.word_embeddings if hasattr(self.model, 'module') else \
                          self.model.roberta.embeddings.word_embeddings
                          )
        if self.args.pgd != 0:
            pass
        os.makedirs(self.args.savedmodel_path, exist_ok=True)
        logging.info("Training/evaluation parameters: %s", args)

        
    def get_model(self):
        raise NotImplementedError('you need implemented this function')
    
    def get_dataloader(self):
        raise NotImplementedError('you need implemented this function')
        
        
    def resume(self):
        if self.args.ckpt_file is not None:

            checkpoint = torch.load(self.args.ckpt_file, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"load resume sucesses! epoch: {self.start_epoch - 1}, mean f1: {checkpoint['mean_f1']}")
        else:
            self.start_epoch = 0
        
        
    def train(self):
        total_step = 0
        best_score = self.args.best_score
        start_time = time.time()
        num_total_steps = len(self.train_dataloader) * (self.args.max_epochs)
        self.optimizer.zero_grad()
        for epoch in range(self.args.max_epochs):
            for single_step, batch in enumerate(tqdm(self.train_dataloader,desc="Training:")):
                self.model.train()
                for key in batch:
                    batch[key] = batch[key].cuda()
                loss,acc,_ = self.model(batch)
                if self.args.distributed_train:
                    loss = loss.mean()
                    acc = acc.mean()
                loss.backward()
                if self.args.fgm !=0 :
                    self.fgm.attack(0.2+epoch * 0.1)
                    loss_adv, _, _ = self.model(batch)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward()
                    self.fgm.restore()
                if self.args.pgd !=0 :
                    pass
                torch.nn.utils.clip_grad_norm_(self.model.cls.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.args.ema_start >= 0 and total_step >= self.args.ema_start:
                    self.ema.update()
                if self.args.swa_start > 0:
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
                else:
                    self.scheduler.step()
                
                total_step += 1
                if total_step % self.args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, total_step)
                    remaining_time = time_per_step * (num_total_steps - total_step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"Epoch {epoch}" 
                                    f"total_step {total_step}" 
                                    f"eta {remaining_time}:" 
                                    f"loss {loss:.3f}, acc {acc:.3f}")
                    
                if total_step % self.args.save_steps == 0:
                    if self.args.ema_start >= 0:
                        self.ema.apply_shadow()
                    loss, result = self.validate()
                    if self.args.ema_start >= 0:
                        self.ema.restore()
                    mean_f1 = result['mean_f1']
                    if mean_f1 > self.args.best_score:
                        state = {
                                'epoch': epoch, 
                                'mean_f1': mean_f1,
                                'optimizer': self.optimizer.state_dict(), 
                                'scheduler': self.scheduler.state_dict(),
                                }
                        if self.args.ema_start >= 0:
                            state['shadow'] = self.ema.shadow,
                            state['backup'] = self.ema.backup,
                        if self.args.distributed_train:
                            if self.args.swa_start > 0:
                                state['model_state_dict'] = self.swa_model.module.state_dict()
                            else:
                                state['model_state_dict'] = self.model.module.state_dict()
                        else:
                            state['model_state_dict'] = self.model.state_dict()
                        torch.save(state, f'{self.args.savedmodel_path}/model_epoch_{epoch}_f1_{mean_f1:.4f}_{total_step}.bin')
                        self.args.best_score = mean_f1
                        logging.info(f"best_score {self.args.best_score}")
                    logging.info(f"current_score {mean_f1}")
                        
            # Validation
            if self.args.ema_start >= 0:
                self.ema.apply_shadow()
            loss, result = self.validate()
            if self.args.ema_start >= 0:
                self.ema.restore()
            mean_f1 = result['mean_f1']
            if mean_f1 > self.args.best_score:
                state = {
                        'epoch': epoch, 
                        'mean_f1': mean_f1,
                        'optimizer': self.optimizer.state_dict(), 
                        'scheduler': self.scheduler.state_dict(),
                        }
                if self.args.ema_start >= 0:
                    state['shadow'] = self.ema.shadow,
                    state['backup'] = self.ema.backup,
                if self.args.distributed_train:
                    if self.args.swa_start > 0:
                        state['model_state_dict'] = self.swa_model.module.state_dict()
                    else:
                        state['model_state_dict'] = self.model.module.state_dict()
                else:
                    state['model_state_dict'] = self.model.state_dict()
                torch.save(state, f'{self.args.savedmodel_path}/model_epoch_{epoch}_f1_{mean_f1:.4f}_{total_step}.bin')
                self.args.best_score = mean_f1
                logging.info(f"best_score {self.args.best_score}")
                
                
    def validate(self):
        self.model.eval()
        if self.args.swa_start > 0:
            torch.optim.swa_utils.update_bn(self.train_dataloader, self.swa_model)
        predictions = []
        labels = []
        losses = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.valid_dataloader,desc="Evaluating")):
                for k in batch:
                    batch[k] = batch[k].cuda()
                if self.args.swa_start > 0:
                    loss, accuracy, pred_label_id = self.swa_model(batch)
                else:
                    loss, accuracy, pred_label_id = self.model(batch)
                loss = loss.mean()
                predictions.extend(pred_label_id.cpu().numpy())
                labels.extend(batch['label'].cpu().numpy())
                losses.append(loss.cpu().numpy())
        loss = sum(losses) / len(losses)
        f1_micro = f1_score(labels, predictions, average='micro')
        f1_macro = f1_score(labels, predictions, average='macro')
        acc=accuracy_score(labels, predictions)
        result = dict(
            accuracy = acc,
            f1_micro = f1_micro,
            f1_macro = f1_macro,
            mean_f1 = (f1_micro+f1_macro)/2.0
        )
        return loss,result
        

                    
                
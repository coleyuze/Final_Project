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
        # Build the optimizer and learning rate scheduler
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
        # restore the model from the checkpoint
        self.resume()
        if args.device == 'cuda':
            if args.distributed_train:# If using distributed training
                print("Multi-gpu training!")
                self.model = torch.nn.parallel.DataParallel(self.model)
        if self.args.fgm != 0:
            logging.info("FGM is used for embedding attack！")
            self.fgm = FGM(self.model.module.roberta.embeddings.word_embeddings if hasattr(self.model, 'module') else \
                          self.model.roberta.embeddings.word_embeddings
                          )
        if self.args.pgd != 0:
            pass
        # create directory to hold models
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

        # Train function to train the model

    def train(self):
        print("basetrian.py==train method")
        total_step = 0  # Initialize total steps
        best_score = self.args.best_score  # Get the best score
        start_time = time.time()  # Start the timer

        # Calculate the total number of steps
        # num_total_steps = len(self.train_dataloader) * (self.args.max_epochs - self.start_epoch)

        num_total_steps = len(self.train_dataloader) * (self.args.max_epochs)
        # Clear the gradients
        self.optimizer.zero_grad()
        # Loop over the epochs
        for epoch in range(self.args.max_epochs):
            print("basetrain.py==train method==开始遍历")
            # 为for循环设置single_step变量，将self.train_dataloader用tqdm模块进行迭代，设置描述文字“Training"
            for single_step, batch in enumerate(tqdm(self.train_dataloader, desc="Training:")):
                self.model.train()  # Set the model to training mode
                for key in batch:
                    batch[key] = batch[key].cuda()  # Move the batch to GPU
                loss, acc, _ = self.model(batch)  # Calculate the loss and accuracy

                # If distributed training is enabled, take the average of the loss and accuracy
                if self.args.distributed_train:
                    loss = loss.mean()
                    acc = acc.mean()
                loss.backward()  # Backward pass to calculate gradients

                # If fast gradient sign method is enabled, perform the attack and calculate the loss again
                if self.args.fgm != 0:
                    self.fgm.attack(0.2 + epoch * 0.1)
                    loss_adv, _, _ = self.model(batch)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward()
                    self.fgm.restore()
                # If projected gradient descent is enabled, perform the attack
                if self.args.pgd != 0:
                    pass  # 后续需要再补充

                # Clip the gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.cls.parameters(), self.args.max_grad_norm)

                self.optimizer.step()  # Update the model parameters
                self.optimizer.zero_grad()  # Clear the gradients
                # If exponential moving average is enabled and total step is greater than or equal to start, update the EMA
                if self.args.ema_start >= 0 and total_step >= self.args.ema_start:
                    self.ema.update()
                # If stochastic weight averaging is enabled, update the SWA model parameters
                if self.args.swa_start > 0:
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()  # Update the learning rate scheduler
                else:
                    self.scheduler.step()

                total_step += 1  # Increment the total steps
                # Print training statistics every print_steps steps
                if total_step % self.args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, total_step)
                    remaining_time = time_per_step * (num_total_steps - total_step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"Epoch {epoch}"
                                 f"total_step {total_step}"
                                 f"eta {remaining_time}:"
                                 f"loss {loss:.3f}, acc {acc:.3f}")
                # Save the model every save_steps steps if the mean f1 score is greater than the current best score
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
                        torch.save(state,
                                   f'{self.args.savedmodel_path}/model_epoch_{epoch}_f1_{mean_f1:.4f}_{total_step}.bin')
                        self.args.best_score = mean_f1
                        logging.info(f"best_score {self.args.best_score}")
                    logging.info(f"current_score {mean_f1}")

            # Validation
            if self.args.ema_start >= 0:
                # The parameters of the current model are applied to the shadow model of EMA
                self.ema.apply_shadow()
            # loss and metrics are calculated on the validation set
            loss, result = self.validate()
            # If the start epoch of the EMA is specified
            if self.args.ema_start >= 0:
                # Recover the parameters of the original model and overwrite the parameters of the EMA shadow model
                self.ema.restore()
            # Get the average F1 value for the current epoch
            mean_f1 = result['mean_f1']
            # If the current average F1 value exceeds the best historical average F1 value
            if mean_f1 > self.args.best_score:
                # Record the current state of the model, including epoch, average F1, optimizer, and learning rate scheduler
                state = {
                    'epoch': epoch,
                    'mean_f1': mean_f1,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }
                # If the start epoch of the EMA is specified
                if self.args.ema_start >= 0:
                    # The state of the EMA shadow model is also recorded
                    state['shadow'] = self.ema.shadow,
                    state['backup'] = self.ema.backup,
                if self.args.distributed_train:
                    # If the start epoch of SWA is specified
                    if self.args.swa_start > 0:
                        # The state of the SWA model is recorded
                        state['model_state_dict'] = self.swa_model.module.state_dict()
                    else:
                        # The state of the original model is recorded
                        state['model_state_dict'] = self.model.module.state_dict()
                else:
                    state['model_state_dict'] = self.model.state_dict()
                # Save the model state to a file
                torch.save(state, f'{self.args.savedmodel_path}/model_epoch_{epoch}_f1_{mean_f1:.4f}_{total_step}.bin')
                self.args.best_score = mean_f1  # Update the historical best average F1 score
                logging.info(f"best_score {self.args.best_score}")  # Output log information

    def validate(self):
        print("basetrain.py==validate")
        self.model.eval()
        # Update the batch normalization statistics of the SWA model (if used)
        if self.args.swa_start > 0:
            torch.optim.swa_utils.update_bn(self.train_dataloader, self.swa_model)
        # Initialize empty lists to store predictions, labels and losses
        predictions = []
        labels = []
        losses = []
        # Disable gradient computation during validation
        with torch.no_grad():
            # Iterate over validation dataloader and evaluate model
            for step, batch in enumerate(tqdm(self.valid_dataloader, desc="Evaluating")):
                for k in batch:  # Move batch to GPU
                    batch[k] = batch[k].cuda()
                # Compute model output (logits), loss and accuracy
                if self.args.swa_start > 0:
                    loss, accuracy, pred_label_id = self.swa_model(batch)
                else:
                    loss, accuracy, pred_label_id = self.model(batch)
                loss = loss.mean()  # Compute mean of loss tensor
                # Append predicted label ids, true label ids and loss to their respective lists
                predictions.extend(pred_label_id.cpu().numpy())
                labels.extend(batch['label'].cpu().numpy())
                losses.append(loss.cpu().numpy())
        # Compute mean of all losses
        loss = sum(losses) / len(losses)
        # results = evaluate(predictions, labels)
        # Compute evaluation metrics using sklearn
        f1_micro = f1_score(labels, predictions, average='micro')
        f1_macro = f1_score(labels, predictions, average='macro')
        acc = accuracy_score(labels, predictions)
        # Store evaluation metrics in a dictionary
        result = dict(
            accuracy=acc,
            f1_micro=f1_micro,
            f1_macro=f1_macro,
            mean_f1=(f1_micro + f1_macro) / 2.0
        )
        # Return loss and evaluation metrics
        return loss, result
        

                    
                
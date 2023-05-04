import torch


# For each x:

# 1. Compute the forward loss for x and backpropagate to get the gradient

# 2. Calculate r from the gradient of the embedding matrix and add it to the current embedding, which is equivalent to x+r

# 3. Compute the forward loss for x+r, backpropagate to get the adversarial gradient, and accumulate to the gradient of (1)

# 4. Restore the embedding to its value at (1)

# 5. Update the parameters according to the gradient of (3)
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.1):
        for name, param in self.model.named_parameters():      
            if param.requires_grad:
                # print(name)

                '''
                Example
                # 初始化
                fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
                for batch_input, batch_label in data:
                    # 正常训练
                    loss = model(batch_input, batch_label)
                    loss.backward() # 反向传播，得到正常的grad
                    # 对抗训练
                    fgm.attack() # 在embedding上添加对抗扰动
                    loss_adv = model(batch_input, batch_label)
                    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore() # 恢复embedding参数
                    # 梯度下降，更新参数
                    optimizer.step()
                    model.zero_grad()
                '''

                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    #The gradient is scaled and then multiplied by an epsilon coefficient to obtain adversarial samples
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD:

    def __init__(self, model, eps=1., alpha=0.3):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
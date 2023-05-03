from .basetrain import BaseTrainer
from src.models import FtModel
from src.config import parse_args
from src.dataset import FinetuneDataset
class FtTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        
    def get_dataloader(self):
        self.train_dataloader, self.valid_dataloader = FinetuneDataset.create_dataloaders(self.args)
        
    def get_model(self):
        self.model = FtModel(self.args)
        
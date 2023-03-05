import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger

from model_loader import LitNetModel
from data_loader import LitFinData
from models.simple_net import SimpleNet
from models.simple_cnn_net import CnnSimpleNet
from LitTorch.prepare_config import df, config
from LitTorch.models.ResCnn import ResCnn
logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = pl.Trainer(

    gpus=0,
    logger=logger,
    # max_epochs=300,
    # benchmark=True,
    # log_every_n_steps=1,
    auto_lr_find=False,
)
config['learning_rate'] =  5e-4

if __name__ == '__main__':
    data_module = LitFinData(df, config)
    model = LitNetModel(ResCnn, config)
    #lr_finder = Tuner(trainer).lr_find(model, train_dataloaders=data_module)
    #model.hparams.lr = lr_finder.suggestion()
    #print(f'Auto-find model LR: {model.hparams.lr}')

    trainer.fit(model, datamodule=data_module)

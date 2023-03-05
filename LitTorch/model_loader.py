import torch
import pytorch_lightning as pl


class LitNetModel(pl.LightningModule, ):

    def __init__(self,
                 net_model,
                 config: dict,
                 ):
        super().__init__()

        # configuration
        self.lr = config['learning_rate']

        # model initialization
        self.nn_model = net_model(config)
        self.l1_loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        # process model
        x = self.nn_model(x)
        # criterion

        loss = torch.sqrt(self.l1_loss(x, train_batch[1], ))
        # logger
        metrics = {'loss': loss, }
        self.log_dict(metrics)
        return metrics

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        # process model
        x = self.nn_model(x)
        # criterion
        loss = torch.sqrt(self.l1_loss(x, val_batch[1], ))
        # logger
        metrics = {'val_loss': loss, }
        print('val_loss', loss)
        self.log_dict(metrics)
        return metrics

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("ptl/val_loss", float(avg_loss.detach().cpu()))





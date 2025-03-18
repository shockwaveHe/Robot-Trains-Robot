import pytorch_lightning as pl
import abc


class BaseSystem(pl.LightningModule, abc.ABC):
    def __init__(self, cfg):
        super(BaseSystem, self).__init__()

    def training_step(self, batch, **kwargs):
        pass

    def build_model(self, **kwargs):
        pass

    def build_loss_func(self):
        pass

    def configure_optimizers(self, **kwargs):
        pass

    def validation_step(self, batch, batch_idx, **kwargs):
        pass

    def test_step(self, batch, batch_idx, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError

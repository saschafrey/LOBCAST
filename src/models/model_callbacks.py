import pytorch_lightning as pl
import src.constants as cst
from pytorch_lightning.callbacks import ModelCheckpoint
from flax.training import train_state,checkpoints
from orbax import checkpoint
from typing import Optional

def callback_save_model(config, run_name):
    monitor_var = config.EARLY_STOPPING_METRIC
    check_point_callback = ModelCheckpoint(
        monitor=monitor_var,
        verbose=True,
        save_top_k=1,
        mode='max',
        dirpath=cst.DIR_SAVED_MODEL + config.WANDB_SWEEP_NAME,
        filename=config.WANDB_SWEEP_NAME + "-run=" + run_name + "-{epoch}-{" + monitor_var + ':.2f}'
    )
    return check_point_callback

def callback_save_model_orbax(config, run_name):
    monitor_var = config.EARLY_STOPPING_METRIC
    check_point_callback = OrbaxModelCheckpoint(
        monitor=monitor_var,
        verbose=True,
        save_top_k=5,
        mode='max',
        dirpath=cst.DIR_SAVED_MODEL + config.WANDB_SWEEP_NAME,
        filename=config.WANDB_SWEEP_NAME + "-run=" + run_name
    )
    return check_point_callback


def early_stopping(config):
    """ Stops if models stops improving. """
    monitor_var = config.EARLY_STOPPING_METRIC
    return pl.callbacks.EarlyStopping(
        monitor=monitor_var,
        min_delta=0.01,
        patience=8,
        verbose=True,
        mode='max',
        # |v stops when if after epoch 1, the
        # check_on_train_epoch_end=True,
        # divergence_threshold=1/3,
    )

from weakref import proxy
from flax.jax_utils import unreplicate
from flax.training import checkpoints
import orbax.checkpoint


class OrbaxModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self,trainer: "pl.Trainer",filepath: str) -> None:
        
        chkpt=self.save_jax_ckpt(trainer,filepath)
        
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    def save_jax_ckpt(self, trainer: "pl.Trainer",filepath: str) -> None:
        pl_ckpt=trainer._checkpoint_connector.dump_checkpoint(self.save_weights_only)
        exclude_keys=[trainer.datamodule.__class__.__qualname__]
        pl_ckpt= {k: pl_ckpt[k] for k in set(list(pl_ckpt.keys())) - set(exclude_keys)}
        with open('scratch_output.txt','w') as f:
            print(pl_ckpt,file=f)
        jax_ckpt = {
            'model': unreplicate(trainer.model.state),
            # Ignore passing any of the config file because haven't found a good way to serialise yet. 
            # 'config': vars(trainer.model.config),
        }
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoints.save_checkpoint(
            ckpt_dir=filepath+"_orbax",
            target=jax_ckpt,
            step=pl_ckpt['epoch'],
            overwrite=True,
            keep=self.save_top_k,
            keep_every_n_steps=10,
            orbax_checkpointer=orbax_checkpointer
        )





    # Default version of _save_checkpoint() lets the Trainer routine run this.
    # Doing it manually now. 
    # def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
    #     trainer.save_checkpoint(filepath, self.save_weights_only)

    #     self._last_global_step_saved = trainer.global_step
    #     self._last_checkpoint_saved = filepath

    #     # notify loggers
    #     if trainer.is_global_zero:
    #         for logger in trainer.loggers:
    #             logger.after_save_checkpoint(proxy(self))


def load_checkpoint_lobcast(
        state: train_state.TrainState,
        path: str,
        step: Optional[int] = None,
    ) -> train_state.TrainState:
    ckpt = {
        'model': state,
    }
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    restored = checkpoints.restore_checkpoint(
        path,
        ckpt,
        step=step,
        orbax_checkpointer=orbax_checkpointer
    )
    return restored
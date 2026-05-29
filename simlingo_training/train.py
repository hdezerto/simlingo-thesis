import os
import hydra

from omegaconf import OmegaConf
import torch
import wandb

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ThroughputMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from transformers import AutoProcessor

from simlingo_training.utils.logging_project import setup_logging, sync_wandb

from simlingo_training.config import TrainConfig
from simlingo_training.callbacks.visualise import VisualiseCallback


@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    # turn off wandb uploading when in debug mode
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"
    
    cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}"
    
    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = None #f"pretrained/{(model_type_name)}"
    temporal_enabled = cfg.model.temporal_model.enabled
    num_temporal_tokens = cfg.model.temporal_model.num_queries if temporal_enabled else 0
    
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        temporal_enabled=temporal_enabled,
        num_temporal_tokens=num_temporal_tokens,
        _recursive_=False
    )
    
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )

    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")

        lora_key_markers = (
            "lora_A",
            "lora_B",
            "lora_embedding_A",
            "lora_embedding_B",
            "lora_magnitude_vector",
        )
        reset_lora = getattr(cfg, "reset_llm_lora_from_checkpoint", False)
        if reset_lora:
            lora_keys = [
                key for key in state_dict
                if any(marker in key for marker in lora_key_markers)
            ]
            state_dict = {key: value for key, value in state_dict.items() if key not in lora_keys}
            print(f"Resetting LLM LoRA: skipped {len(lora_keys)} LoRA checkpoint tensors.")

        strict_checkpoint_loading = not (
            cfg.model.temporal_model.enabled or reset_lora
        )
        load_result = model.load_state_dict(state_dict, strict=strict_checkpoint_loading)
        if not strict_checkpoint_loading:
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
            temporal_prefixes = ("temporal_encoder.", "temporal_motion_head.")

            def expected_missing_key(key):
                if key.startswith(temporal_prefixes):
                    return True
                if reset_lora and any(marker in key for marker in lora_key_markers):
                    return True
                return False

            non_temporal_missing = [key for key in missing_keys if not expected_missing_key(key)]
            non_temporal_unexpected = [key for key in unexpected_keys if not key.startswith(temporal_prefixes)]
            if non_temporal_missing or non_temporal_unexpected:
                raise RuntimeError(
                    "Checkpoint loading failed outside the expected newly initialised modules. "
                    f"Missing keys: {non_temporal_missing}. "
                    f"Unexpected keys: {non_temporal_unexpected}."
                )
            if missing_keys:
                print("Initialising expected new weights from scratch:")
                for key in missing_keys:
                    print(f"  - {key}")

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"
    
    overfit = cfg.overfit if cfg.overfit > 0 else 0
        
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = cfg.resume_path
    resume_wandb = False

    # if folder for this experiment does not exist set resume to true
    # to create necessary folders to resume wandb logging later
    if resume_path is not None and not os.path.exists(resume_path):
        resume_wandb = True
    elif resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_wandb = True

    if resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None

    # setup lightning logger
    loggers = []
    # csvlogger = CSVLogger("log/", "CSVLogger")
    # loggers.append(csvlogger)
    # csvlogger = None

    wandblogger = WandbLogger(
        project=cfg.wandb_project,
        id=cfg.wandb_name,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        resume=resume_wandb,
    )
    wandblogger.watch(model)
    loggers.append(wandblogger)

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        monitor=None,
        dirpath="./checkpoints",
        filename="{epoch:03d}",
        save_last=False,
        every_n_epochs=cfg.val_every_n_epochs,
        # every_n_train_steps=cfg.val_check_interval,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=3)
    callbacks=[
        checkpoint_callback, 
        model_summary, 
        # ThroughputMonitor(batch_size_fn=lambda batch: batch.driving_input.camera_images.size(0)), 
        VisualiseCallback(interval=1000, val_interval=1000)
    ]
    if not cfg.debug: 
        callbacks.append(lr_monitor)
    
    print(f"Number of GPUS: {cfg.gpus}")
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=callbacks,
            devices=cfg.gpus,
            # enable_checkpointing=False,
            gradient_clip_val=0.3,
            # gradient_clip_algorithm="value",
            # log_every_n_steps=10,
            logger=loggers,
            # max_steps=cfg.max_steps,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            # use_distributed_sampler=False,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
            # val_check_interval=cfg.val_check_interval,
        )

    trainer.fit(model, data_module, ckpt_path=resume_path)

    # Keep the periodic epoch checkpoints for recovery/comparison, but also
    # write an explicit final checkpoint so the last trained state is easy to
    # identify even when max_epochs is not aligned with every_n_epochs.
    final_checkpoint_path = os.path.join("checkpoints", "final.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)
    if trainer.is_global_zero:
        print(f"Saved final checkpoint to {final_checkpoint_path}")

    wandb.finish()

if __name__ == "__main__":
    main()

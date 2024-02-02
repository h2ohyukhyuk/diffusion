# https://github.com/spmallick/learnopencv/blob/master/Guide-to-training-DDPMs-from-Scratch/Generating_MNIST_using_DDPMs.ipynb
import torch
import torch.nn as nn
from torch.cuda import amp

from torchmetrics import MeanMetric

import os
from utils import get_default_device, setup_log_directory
from dataclasses import dataclass

from dataset import get_dataloader
from simple_diffusion import SimpleDiffusion, forward_diffusion, reverse_diffusion

from tqdm import tqdm
from unet import UNet
import gc
from datetime import datetime

@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "MNIST"  # "MNIST", "Cifar-10", "Cifar-100", "Flowers"

    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join("Logs_Checkpoints", "Inference")
    root_checkpoint_dir = os.path.join("Logs_Checkpoints", "checkpoints")

    # Current log and checkpoint directory.
    log_dir = "version_0"
    checkpoint_dir = "version_0"


@dataclass
class TrainingConfig:
    TIMESTEPS = 1000  # Define number of diffusion timesteps
    IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32)
    NUM_EPOCHS = 30
    BATCH_SIZE = 128
    LR = 2e-4
    NUM_WORKERS = 2

@dataclass
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 512
    BASE_CH_MULT = (1, 2, 4, 8) # 32, 16, 8, 4
    APPLY_ATTENTION = (False, False, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 2 # 128

def train_one_epoch(model, sd, loader, optimizer, scaler, loss_fn, epoch=800,
                    base_config=BaseConfig(), training_config=TrainingConfig()):
    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")

        for x0s, _ in loader:
            tq.update(1)

            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = forward_diffusion(sd, x0s, ts)

            with amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()

        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss

def train():
    loader = get_dataloader(
        dataset_name=BaseConfig.DATASET,
        batch_size=128,
        device='cpu',
    )

    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    model.to(BaseConfig.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

    dataloader = get_dataloader(
        dataset_name  = BaseConfig.DATASET,
        batch_size    = TrainingConfig.BATCH_SIZE,
        device        = BaseConfig.DEVICE,
        pin_memory    = True,
        num_workers   = TrainingConfig.NUM_WORKERS,
    )

    loss_fn = nn.MSELoss()

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    scaler = amp.GradScaler()

    total_epochs = TrainingConfig.NUM_EPOCHS + 1
    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

    generate_video = False
    ext = ".mp4" if generate_video else ".png"

    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        # Algorithm 1: Training
        train_one_epoch(model, sd, dataloader, optimizer, scaler, loss_fn, epoch=epoch)

        if epoch % 5 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")

            # Algorithm 2: Sampling
            reverse_diffusion(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=32,
                              generate_video=generate_video,
                              save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE,
                              )

            # clear_output()
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict


def inference():

    model = UNet(
        input_channels=TrainingConfig.IMG_SHAPE[0],
        output_channels=TrainingConfig.IMG_SHAPE[0],
        base_channels=ModelConfig.BASE_CH,
        base_channels_multiples=ModelConfig.BASE_CH_MULT,
        apply_attention=ModelConfig.APPLY_ATTENTION,
        dropout_rate=ModelConfig.DROPOUT_RATE,
        time_multiple=ModelConfig.TIME_EMB_MULT,
    )
    # checkpoint_dir = "/kaggle/working/Logs_Checkpoints/checkpoints/version_0"

    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])

    model.to(BaseConfig.DEVICE)

    sd = SimpleDiffusion(
        num_diffusion_timesteps=TrainingConfig.TIMESTEPS,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
    )

    log_dir = "inference_results"
    os.makedirs(log_dir, exist_ok=True)

    generate_video = True

    ext = ".mp4" if generate_video else ".png"
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"

    save_path = os.path.join(log_dir, filename)

    reverse_diffusion(
        model,
        sd,
        save_path=save_path,
        num_images=64,
        generate_video=generate_video,
        timesteps=1000,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
        nrow=8,
    )
    print(save_path)

if __name__ == "__main__":
    train()
    # inference()
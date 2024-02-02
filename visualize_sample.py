from dataset import get_dataloader
import matplotlib.pyplot as plt
from utils import  inverse_transform
from torchvision.utils import make_grid
from simple_diffusion import SimpleDiffusion, forward_diffusion
import torch

DATASET = "MNIST"
TIMESTEPS = 1000

loader = get_dataloader(
    dataset_name=DATASET,
    batch_size=128,
    device='cpu',
)

sd = SimpleDiffusion(num_diffusion_timesteps=TIMESTEPS, device="cpu")

plt.figure(figsize=(12, 6), facecolor='white')

for b_image, _ in loader:
    b_image = inverse_transform(b_image).cpu()
    grid_img = make_grid(b_image / 255.0, nrow=16, padding=True, pad_value=1, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    break

plt.savefig(f'debug/{DATASET}_samples.png')

loader = iter(  # converting dataloader into an iterator for now.
    get_dataloader(
        dataset_name=DATASET,
        batch_size=6,
        device="cpu",
    )
)

x0s, _ = next(loader)
noisy_images = []
specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

for timestep in specific_timesteps:
    timestep = torch.as_tensor(timestep, dtype=torch.long)

    xts, _ = forward_diffusion(sd, x0s, timestep)
    xts = inverse_transform(xts) / 255.0
    xts = make_grid(xts, nrow=1, padding=1)

    noisy_images.append(xts)

# Plot and see samples at different timesteps

_, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor='white')

for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
    ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
    ax[i].set_title(f"t={timestep}", fontsize=8)
    ax[i].axis("off")
    ax[i].grid(False)

plt.suptitle("Forward Diffusion Process", y=0.9)
plt.axis("off")
#plt.show()
plt.savefig(f'debug/{DATASET}_forward_diffustion.png')
import torch
from dataset import ImageDataset
from op import SuperResolutionOperator
from torchvision import transforms
import numpy as np
from diffusers import StableDiffusion3Pipeline, AutoencoderKL
import os
from torchvision.utils import save_image
from taming.modules.losses.lpips import LPIPS

def cal_psnr(a,b):
    a = torch.round(((a + 1.0) / 2) * 255.0)
    b = torch.round(((b + 1.0) / 2) * 255.0)
    mse = torch.mean((a-b)**2)
    return 10 * torch.log10(255**2 / mse)

def print_avgs(avgs, a=0):
    for key in avgs.keys():
        print("{0}: {1:.4}, ".format(key, np.mean(avgs[key][a:])), end="")
    print("")

if __name__ == "__main__":
    seed=64
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed=seed)

    DTYPE = torch.float32
    DATA_ROOT = "./imagenet_512"
    OUT_ROOT = "./results/imagenet/savi"
    SAVI_STEP = 100

    out_dirs = ["source", "favi", "savi"]
    out_dirs = [os.path.join(OUT_ROOT, o) for o in out_dirs]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)

    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageDataset(root=DATA_ROOT, transform=test_transforms, return_path=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="vae",
    )

    vae.to("cuda")
    perceptual_loss = LPIPS().eval().to("cuda")

    avgs = {
        "psnr_favi": [], "psnr_savi": [],
        "lpips_favi": [], "lpips_savi": [],
        "kl_favi": [], "kl_savi": [],
    }
    for i, (x, x_path) in enumerate(dataloader):

        x = x.to(dtype=DTYPE, device="cuda")
        x_name = x_path[0].split('/')[-1]

        with torch.no_grad():
            posterior = vae.encode(x).latent_dist
        
        z_sample = posterior.mean + torch.randn_like(posterior.std) * posterior.std
        x_favi = vae.decode(z_sample).sample

        posterior.mean = posterior.mean.requires_grad_()
        posterior.std = posterior.std.requires_grad_()
        opt = torch.optim.Adam([posterior.mean, posterior.std], lr=1e-2)

        for j in range(SAVI_STEP):
            z_sample = posterior.mean + torch.randn_like(posterior.std) * posterior.std
            x_hat = vae.decode(z_sample).sample
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            rec_loss = (x - x_hat)**2
            p_loss = perceptual_loss(x, x_hat)
            rec_loss = rec_loss + 0.1 * p_loss
            nll_loss = rec_loss
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            loss = 1.0 * kl_loss + nll_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            psnr = cal_psnr(x, x_hat)

            if j == 0:
                print("[init] psnr: {0:.4f}, rate: {1:.4f}, lpips: {2:.4f}, loss: {3:.4f}".format(psnr, kl_loss, p_loss.item(), loss))
                avgs["psnr_favi"].append(psnr.item())
                avgs["lpips_favi"].append(p_loss.item())
                avgs["kl_favi"].append(kl_loss.item())
            if j + 1 == SAVI_STEP:
                print("[term] psnr: {0:.4f}, rate: {1:.4f}, lpips: {2:.4f}, loss: {3:.4f}".format(psnr, kl_loss, p_loss.item(), loss))
                avgs["psnr_savi"].append(psnr.item())
                avgs["lpips_savi"].append(p_loss.item())
                avgs["kl_savi"].append(kl_loss.item())
                x_savi = x_hat

        out_tensors = [x, x_favi, x_savi]

        for i in range(3):
            save_image(out_tensors[i], os.path.join(out_dirs[i], x_name), normalize=True, value_range=(-1, 1))

    print_avgs(avgs)

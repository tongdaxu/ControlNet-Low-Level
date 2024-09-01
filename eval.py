from dataset import ImageDataset
import torch
from torchvision import transforms
from fid import fid_pytorch, cal_psnr
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import lpips
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure

def print_avgs(avgs, a=0):
    for key in avgs.keys():
        print("{0}: {1:.4}, ".format(key, np.mean(avgs[key][a:])), end="")
    print("")

ref_path = '/NEW_EDS/JJ_Group/xutd/diffusion-inversion/results/imagenet/srx8/max_500s_4.8/source'
dis_path = '/NEW_EDS/JJ_Group/xutd/diffusion-inversion/results/imagenet/srx8/max_500s_4.8/recon'

test_transforms = transforms.Compose(
    [transforms.ToTensor()]
)
ref_dataset = ImageDataset(root=ref_path, transform=test_transforms, return_path=True)
ref_dataloader = torch.utils.data.DataLoader(ref_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
ref_dataloader = list(ref_dataloader)
dis_dataset = ImageDataset(root=dis_path, transform=test_transforms, return_path=True)
dis_dataloader = torch.utils.data.DataLoader(dis_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
dis_dataloader = list(dis_dataloader)

fid_computer = fid_pytorch()
kid_computer = KernelInceptionDistance(subsets=100, subset_size=10, normalize=True).cuda()
fid_patch = 512
ssim_computer = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

avgs = {
    "mse": [], "psnr": [],
    "fid": [], "kid": [], "kid_std": [], "lpips": [], "ssim": [],
}

with torch.no_grad():
    fid_computer.clear_pools()
    for i, ((x, _), (x_hat, _)) in tqdm(enumerate(zip(ref_dataloader, dis_dataloader))):
        if i >= 30:
            break
        x1 = x.cuda()
        x2 = x_hat.cuda()
        
        unfold = nn.Unfold(kernel_size=(fid_patch, fid_patch),stride=(fid_patch, fid_patch))
        x1_unfold = unfold(x1).reshape(1, 3, fid_patch, fid_patch, -1)
        x1_unfold = torch.permute(x1_unfold, (0, 4, 1, 2, 3)).reshape(-1, 3, fid_patch, fid_patch)
        x2_unfold = unfold(x2).reshape(1, 3, fid_patch, fid_patch, -1)
        x2_unfold = torch.permute(x2_unfold, (0, 4, 1, 2, 3)).reshape(-1, 3, fid_patch, fid_patch)
        
        fid_computer.add_ref_img(x1_unfold)
        fid_computer.add_dis_img(x2_unfold)

        kid_computer.update(x1_unfold, real=True)
        kid_computer.update(x2_unfold, real=False)

        avgs['mse'].append(torch.mean((x1 - x2)**2).item())
        avgs['psnr'].append(cal_psnr(x1, x2))
        avgs['lpips'].append(loss_fn_alex(x1 * 2.0 - 1.0, x2 * 2.0 - 1.0).item())
        avgs['ssim'].append(ssim_computer(x1, x2).item())

    kid_mean, kid_std = kid_computer.compute()
    avgs['kid'].append(kid_mean.cpu())
    avgs['kid_std'].append(kid_std.cpu())
    avgs['fid'].append(fid_computer.summary_pools())
    print_avgs(avgs)
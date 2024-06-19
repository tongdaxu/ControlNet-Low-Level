import torch
from pipe import StableDiffusionInverse, EulerAncestralDiscreteInverse
from dataset import ImageDataset
from op import SuperResolutionOperator
from torchvision import transforms
import numpy as np

if __name__ == "__main__":
    seed=64
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed=seed)

    DTYPE = torch.float32
    DATA_ROOT = "./ffhq_512"
    
    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageDataset(root=DATA_ROOT, transforms=test_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    f = SuperResolutionOperator([1, 3, 512, 512], 8)
    f = f.to(dtype=DTYPE, device="cuda")

    model_id = "stabilityai/stable-diffusion-2-base"

    scheduler = EulerAncestralDiscreteInverse.from_pretrained(model_id, subfolder="scheduler")

    pipe = StableDiffusionInverse.from_pretrained(model_id, scheduler=scheduler, torch_dtype=DTYPE)
    pipe = pipe.to("cuda")

    for i, x in enumerate(dataloader):
        x = x.to(dtype=DTYPE, device="cuda")
        y = f(x)

        image = pipe(f=f,
                     y=y,
                     scale=2.4,
                     prompt="",
                     height=512,
                     width=512,
                     num_inference_steps=250,
                     guidance_scale=0.0).images[0]  

        image.save("test_sd2.png")
        assert(0)

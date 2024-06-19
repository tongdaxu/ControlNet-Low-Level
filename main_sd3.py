import torch
from pipe import StableDiffusion3Inverse
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
    DATA_ROOT = "./gen_512"

    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageDataset(root=DATA_ROOT, transforms=test_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    f = SuperResolutionOperator([1, 3, 512, 512], 8)
    f = f.to(dtype=DTYPE, device="cuda")

    pipe = StableDiffusion3Inverse.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=DTYPE,
    )
    pipe.to("cuda")

    prompt_dict = torch.load("prompt_dict.pt")

    for i, x in enumerate(dataloader):
        x = x.to(dtype=DTYPE, device="cuda")
        y = f(x)
        image = pipe(
            f=f,
            y=y,
            prompt_embeds=prompt_dict["prompt_embeds"].to(DTYPE),
            negative_prompt_embeds=prompt_dict["negative_prompt_embeds"].to(DTYPE),
            pooled_prompt_embeds=prompt_dict["pooled_prompt_embeds"].to(DTYPE),
            negative_pooled_prompt_embeds=prompt_dict["negative_pooled_prompt_embeds"].to(DTYPE),
            num_inference_steps=250,
            height=512,
            width=512,
            guidance_scale=0.0,
        ).images[0]

        image.save("test_zyr_4_0_250_step.png")
        assert(0)
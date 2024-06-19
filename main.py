import torch
from pipe import StableDiffusion3Inverse

if __name__ == "__main__":

    SAVE_PROMPT = False
    DTYPE = torch.float32

    if SAVE_PROMPT:
        ## save memory
        pipe = StableDiffusion3Inverse.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16
        )
    else:
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

    if SAVE_PROMPT:
        (   
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt = "A high quality photo",
            prompt_2 = "",
            prompt_3 = "",
            negative_prompt = "",
        )
        prompt_dict = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }
        torch.save(prompt_dict, "prompt_dict.pt")

    prompt_dict = torch.load("prompt_dict.pt")

    image = pipe(
        prompt_embeds=prompt_dict["prompt_embeds"].to(DTYPE),
        negative_prompt_embeds=prompt_dict["negative_prompt_embeds"].to(DTYPE),
        pooled_prompt_embeds=prompt_dict["pooled_prompt_embeds"].to(DTYPE),
        negative_pooled_prompt_embeds=prompt_dict["negative_pooled_prompt_embeds"].to(DTYPE),
        num_inference_steps=100,
        height=512,
        width=512,
        guidance_scale=7.0,
    ).images[0]

    image.save("test.png")
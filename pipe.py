import torch
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from types import MethodType

from diffusers import StableDiffusion3Pipeline
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.utils import BaseOutput
from diffusers.configuration_utils import register_to_config
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput 
from diffusers import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteSchedulerOutput
import numpy as np

class EulerAncestralDiscreteInverse(EulerAncestralDiscreteScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False
    ):
        super().__init__(
            num_train_timesteps = num_train_timesteps,
            beta_start = beta_start,
            beta_end = beta_end,
            beta_schedule = beta_schedule,
            trained_betas = trained_betas,
            prediction_type = prediction_type,
            timestep_spacing = timestep_spacing,
            steps_offset = steps_offset,
            rescale_betas_zero_snr = rescale_betas_zero_snr,
        )

    def step(
        self,
        f,
        y,
        scale,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:

        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)

        prev_sample = prev_sample + noise * sigma_up

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        return prev_sample, pred_original_sample, sigma

class StableDiffusionInverse(StableDiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
        ):
        super().__init__(
            vae = vae,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            unet = unet,
            scheduler = scheduler,
            safety_checker = safety_checker,
            feature_extractor = feature_extractor,
            image_encoder = image_encoder,
            requires_safety_checker = requires_safety_checker,
        )

    @torch.no_grad()
    def __call__(
        self,
        f,
        y,
        scale,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                with torch.enable_grad():

                    latents.requires_grad = True
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_next, pred_z0, sigma = self.scheduler.step(f, y, scale, noise_pred, t, latents, **extra_step_kwargs)
                    pred_x0 = self.vae.decode(pred_z0 / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                    # print(torch.min(pred_x0), torch.max(pred_x0))
                    norm = torch.linalg.norm(f(pred_x0) - y)
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]
                    print("distance: {:.4f}".format(norm.item()))

                latents_next = latents_next - norm_grad * scale * sigma
                latents = latents_next.detach()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0]

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

@dataclass
class InverseFlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    denoised: torch.FloatTensor

def step_inverse(
    self,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    stochastic = False,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
) -> Union[InverseFlowMatchEulerDiscreteSchedulerOutput, Tuple]:

    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    sigma = self.sigmas[self.step_index]

    gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

    noise = randn_tensor(
        model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
    )

    eps = noise * s_noise
    sigma_hat = sigma * (gamma + 1)

    if gamma > 0:
        sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

    # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    # NOTE: "original_sample" should not be an expected prediction_type but is left in for
    # backwards compatibility

    # if self.config.prediction_type == "vector_field":
    denoised = sample - model_output * sigma

    # 2. Convert to an ODE derivative
    derivative = (sample - denoised) / sigma_hat

    dt = self.sigmas[self.step_index + 1] - sigma_hat
    ft = - sample / (1 - sigma + 1e-5)
    gt2d2 = sigma / (1 - sigma + 1e-5)
    score = (derivative - ft) / (-gt2d2)
    ex0xt = (sample + sigma**2 * score) / (1 - sigma + 1e-5)

    if stochastic:
        # use euler SDE solver
        # need to recompute all the shit above clear
        prev_sample = sample + (ft - 2 * gt2d2 * score) * dt + noise * torch.sqrt(-dt * 2 * gt2d2)
    else:
        # use euler ODE solver
        prev_sample = sample + derivative * dt
        # prev_sample = sample + (ft - gt2d2 * score) * dt

    # Cast sample back to model compatible dtype
    prev_sample = prev_sample.to(model_output.dtype)
    denoised = denoised.to(model_output.dtype)

    # upon completion increase step index by one
    self._step_index += 1

    if not return_dict:
        return (prev_sample,)

    return InverseFlowMatchEulerDiscreteSchedulerOutput(
        prev_sample=prev_sample,
        denoised=denoised,
    )

class StableDiffusion3Inverse(StableDiffusion3Pipeline):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: None,
        tokenizer: None,
        text_encoder_2: None,
        tokenizer_2: None,
        text_encoder_3: None,
        tokenizer_3: None,
    ):
        super().__init__(
            transformer = transformer,
            scheduler = scheduler,
            vae = vae,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            text_encoder_2 = text_encoder_2,
            tokenizer_2 = tokenizer_2,
            text_encoder_3 = text_encoder_3,
            tokenizer_3 = tokenizer_3,
        )
        self.scheduler.step_inverse = MethodType(step_inverse, self.scheduler)
        self.scale = 4.0
        
    @torch.no_grad()
    def __call__(
        self,
        f,
        y,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                with torch.enable_grad():
                    latents.requires_grad = True

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    sigma = self.scheduler.sigmas[i]
                    scheduler_out = self.scheduler.step_inverse(noise_pred, t, latents, return_dict=True)
                    denoised = scheduler_out.denoised
                    denoised = (denoised / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    denoised = self.vae.decode(denoised, return_dict=False)[0]
                    print(torch.min(denoised), torch.max(denoised))
                    norm = torch.linalg.norm(f(denoised) - y)
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]
                    print("distance: {:.4f}".format(norm.item()))

                latents_next = scheduler_out.prev_sample
                latents_next = latents_next - norm_grad * self.scale * sigma

                latents = latents_next.detach()

                if i % 50 == 0:
                    denoised_image = self.image_processor.postprocess(denoised, output_type=output_type)[0]
                    # denoised_image.save("test_x_pred_{}.png".format(i))
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
    
    
    
    
class StableDiffusion3DSGInverse(StableDiffusion3Pipeline):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: None,
        tokenizer: None,
        text_encoder_2: None,
        tokenizer_2: None,
        text_encoder_3: None,
        tokenizer_3: None,
    ):
        super().__init__(
            transformer = transformer,
            scheduler = scheduler,
            vae = vae,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            text_encoder_2 = text_encoder_2,
            tokenizer_2 = tokenizer_2,
            text_encoder_3 = text_encoder_3,
            tokenizer_3 = tokenizer_3,
        )
        self.scheduler.step_inverse = MethodType(step_inverse, self.scheduler)
        self.scale = 3.0
        
    @torch.no_grad()
    def __call__(
        self,
        f,
        y,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                with torch.enable_grad():
                    latents.requires_grad = True

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    sigma = self.scheduler.sigmas[i]
                    scheduler_out = self.scheduler.step_inverse(noise_pred, t, latents, return_dict=True)
                    denoised = scheduler_out.denoised
                    denoised = (denoised / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    denoised = self.vae.decode(denoised, return_dict=False)[0]
                    print(torch.min(denoised), torch.max(denoised))
                    norm = torch.linalg.norm(f(denoised) - y)
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]
                    print("distance: {:.4f}".format(norm.item()))

                latents_next = scheduler_out.prev_sample
                latents_next = latents_next - norm_grad * self.scale * sigma

                latents = latents_next.detach()

                if i % 50 == 0:
                    denoised_image = self.image_processor.postprocess(denoised, output_type=output_type)[0]
                    denoised_image.save("test_x_pred_zyr_{}.png".format(i))
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
            latents_copy = latents.clone()
            for latent_optimize_time in range(50):
                latent_copy = latents_copy.detach()
                with torch.enable_grad():
                    latent_copy.requires_grad = True
                    latent_copy = (latent_copy / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(latent_copy, return_dict=False)[0]    
                    norm = torch.linalg.norm(f(image) - y)
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=latent_copy)[0]
                    latent_copy = latent_copy - norm_grad
                print("distance: {:.4f}".format(norm.item()))
        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
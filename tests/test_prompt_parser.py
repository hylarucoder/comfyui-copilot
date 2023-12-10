import json
import os

from prompt_parser import parse_workflow

current_path = os.path.dirname(os.path.abspath(__file__))


def test_prompt_parser_02():
    path = f"{current_path}/examples/02.json"
    prompt_info = open(path, "rt").read()
    parsed = parse_workflow(json.loads(prompt_info))
    assert parsed == [
        "CLIPTextEncode->clip->LoraLoader->lora_name::oc_v10.safetensors",
        "CLIPTextEncode->clip->LoraLoader->strength_clip::0.5",
        "CLIPTextEncode->clip->LoraLoader->strength_model::0.5",
        "ImageUpscaleWithModel->upscale_model->UpscaleModelLoader->model_name::RealESRGAN_x4plus_anime_6B.pth",
        "KSampler->latent_image->EmptyLatentImage->batch_size::1",
        "KSampler->latent_image->EmptyLatentImage->height::512",
        "KSampler->latent_image->EmptyLatentImage->width::768",
        "KSampler->model->LoraLoader->lora_name::oc_v10.safetensors",
        "KSampler->model->LoraLoader->strength_clip::0.5",
        "KSampler->model->LoraLoader->strength_model::0.5",
        "KSampler->negative->CLIPTextEncode->text::embeddings:EasyNegative, (worst quality, low quality, normal quality, bad anatomy:1.4), text, watermark,",
        "KSampler->positive->CLIPTextEncode->text::1girl, solo, expressionless, silver hair, ponytail, purple eyes, face mask, ninja, blue bikini, medium breats, cleavage, scarf,",
        "LoraLoader->clip->CheckpointLoaderSimple->ckpt_name::agelesnate_v121.safetensors",
        "LoraLoader->model->CheckpointLoaderSimple->ckpt_name::agelesnate_v121.safetensors",
        "VAEDecode->samples->KSampler->cfg::1.5",
        "VAEDecode->samples->KSampler->cfg::7",
        "VAEDecode->samples->KSampler->denoise::0.5",
        "VAEDecode->samples->KSampler->denoise::1",
        "VAEDecode->samples->KSampler->sampler_name::dpmpp_2m_sde",
        "VAEDecode->samples->KSampler->scheduler::karras",
        "VAEDecode->samples->KSampler->seed::804645408561758",
        "VAEDecode->samples->KSampler->steps::20",
        "VAEDecode->samples->KSampler->steps::40",
        "VAEDecode->vae->VAELoader->vae_name::vae-ft-mse-840000-ema-pruned.safetensors",
        "VAEEncode->pixels->ImageScaleBy->scale_by::0.5",
        "VAEEncode->pixels->ImageScaleBy->upscale_method::area",
        "VAEEncode->vae->VAELoader->vae_name::vae-ft-mse-840000-ema-pruned.safetensors",
    ]

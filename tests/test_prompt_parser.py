import glob
import json
from comfyui_eagle_imageinfo.prompt_parser import parse_prompt_info, parse_workflow


def test_prompt_parser_01():
    path = "./tests/examples/01.json"
    prompt_info = open(path, "rt").read()
    parsed = parse_prompt_info(json.loads(prompt_info))
    # __import__("pdb").set_trace()
    assert parsed == [
        "KSampler:Model:AWPainting 1.1_v1.1.safetensors",
        "KSampler:Negative:(worst quality, low quality:1.4), EasyNegative, ng_deepnegative_v1_75t, bad_prompt_version2,",
        "KSampler:Positive:masterpiece, best quality,\\n1girl, solo, long hair, black hair, from behind",
        "KSampler:cfg:8.0",
        "KSampler:denoise:1.0",
        "KSampler:seed:979537337409677",
        "KSampler:steps:30",
    ]


def test_prompt_parser_02():
    path = "./tests/examples/02.json"
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


def test_prompt_parser_03():
    path = "./tests/examples/03-SDXL.json"
    prompt_info = open(path, "rt").read()
    parsed = parse_prompt_info(json.loads(prompt_info))
    assert parsed == [
        "KSampler:Model:sdXL_v10VAEFix.safetensors",
        "KSampler:Model:sd_xl_refiner_1.0.safetensors",
        "KSampler:Negative:prompt: text, watermark, low quality, medium quality, blurry, censored, wrinkles, deformed, mutated text, watermark, low quality, medium quality, blurry, censored, wrinkles, deformed, mutated",
        "KSampler:Positive:photo of beautiful age 18 girl, pastel hair, freckles sexy, beautiful, close up, young, dslr, 8k, 4k, ultrarealistic, realistic, natural skin, textured skin",
        "KSampler:cfg:8",
        "KSampler:denoise:0.3000000000000001",
        "KSampler:denoise:1",
        "KSampler:seed:0",
        "KSampler:seed:234288240797552",
        "KSampler:steps:20",
        "KSampler:steps:40",
    ]

import gradio as gr
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from PIL import Image
from typing import Optional
import os

import random
import numpy as np
import spaces
import huggingface_hub
import copy

from FlowEdit_utils import FlowEditSD3, FlowEditFLUX
SD3STRING = 'stabilityai/stable-diffusion-3-medium-diffusers'
FLUXSTRING = 'black-forest-labs/FLUX.1-dev'
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# model_type = 'SD3'

pipe_sd3 = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, token=os.getenv('HF_ACCESS_TOK'))
pipe_flux = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16, token=os.getenv('HF_ACCESS_TOK'))
# pipe_sd3.to(device)
# pipe_flux.to(device)

# scheduler = pipe.scheduler
# pipe = pipe.to(device)
loaded_model = 'None'


def on_model_change(model_type):
    if model_type == 'SD3':

        T_steps_value = 50

        src_guidance_scale_value = 3.5

        tar_guidance_scale_value = 13.5

        n_max_value = 33

    elif model_type == 'FLUX':

        T_steps_value = 28

        src_guidance_scale_value = 1.5

        tar_guidance_scale_value = 5.5

        n_max_value = 24

    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    return T_steps_value, src_guidance_scale_value, tar_guidance_scale_value, n_max_value

def get_examples():
    case = [
        ["inputs/cat.png", "SD3", 50,  3.5, 13.5, 33, "A small, fluffy kitten sitting in a grassy field. The kitten is positioned in the center of the scene, surrounded by a field. The kitten appears to be looking at something in the field.", "A small puppy sitting in a grassy field. The puppy is positioned in the center of the scene, surrounded by a field. The puppy appears to be looking at something in the field.", 0, 1, 42, "example_outs/cat_puppy_sd3.png"],
        ["inputs/iguana.png", "SD3", 50,  3.5, 13.5, 31, "A large orange lizard sitting on a rock near the ocean. The lizard is positioned in the center of the scene, with the ocean waves visible in the background. The rock is located close to the water, providing a picturesque setting for the lizard''s resting spot.", "A large dragon sitting on a rock near the ocean. The dragon is positioned in the center of the scene, with the ocean waves visible in the background. The rock is located close to the water, providing a picturesque setting for the dragon''s resting spot.", 0, 1, 42, "example_outs/iguana_dragon.png"],
        ["inputs/cat.png", "FLUX", 28,  1.5, 5.5, 23, "A small, fluffy kitten sitting in a grassy field. The kitten is positioned in the center of the scene, surrounded by a field. The kitten appears to be looking at something in the field.", "A small puppy sitting in a grassy field. The puppy is positioned in the center of the scene, surrounded by a field. The puppy appears to be looking at something in the field.", 0, 1, 42, "example_outs/cat_puppy_flux.png"],
        ["inputs/gas_station.png", "FLUX", 28,  1.5, 5.5, 23, "A gas station with a white and red sign that reads \"CAFE\" There are several cars parked in front of the gas station, including a white car and a van.", "A gas station with a white and red sign that reads \"LOVE\" There are several cars parked in front of the gas station, including a white car and a van.", 0, 1, 42, "example_outs/gas_cafe_love.png"],
        ["inputs/steak.png", "FLUX", 28,  1.5, 5.5, 23, "A steak accompanied by a side of leaf salad.", "A bread roll accompanied by a side of leaf salad.", 0, 1, 42, "example_outs/steak_bread.png"],
        ["inputs/kill_bill.png", "FLUX", 28,  2.5, 6.5, 22, "a blonde woman in a yellow jumpsuit holding a sword in front of her face", "a blonde woman in a yellow jumpsuit holding a sword in front of her face, anime style drawing", 14, 1, 42, "example_outs/kill_bill_anime.png"],

    ]
    return case


@spaces.GPU(duration=75)
def FlowEditRun(
    image_src: str,
    model_type: str,
    T_steps: int,
    src_guidance_scale: float,
    tar_guidance_scale: float,
    n_max: int,
    src_prompt: str,
    tar_prompt: str,
    n_min: int,
    n_avg: int,
    seed: int,

    ):

    if not len(src_prompt):
        raise gr.Error("source prompt cannot be empty")
    if not len(tar_prompt):
        raise gr.Error("target prompt cannot be empty")

    # global pipe_sd3
    # global scheduler
    # global loaded_model

    # reload model only if different from the loaded model
    # if loaded_model != model_type:

    if model_type == 'FLUX':
        # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16, token=os.getenv('HF_ACCESS_TOK'))
        pipe = pipe_flux.to(device) 
    elif model_type == 'SD3':
        # pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, token=os.getenv('HF_ACCESS_TOK'))
        pipe = pipe_sd3.to(device) 
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    scheduler = pipe.scheduler
        # pipe = pipe.to(device)




    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # load image
    image = Image.open(image_src)
    # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    image_src = pipe.image_processor.preprocess(image)
    # image_tar = pipe.image_processor.postprocess(image_src)
    # return image_tar[0]

    # cast image to half precision
    image_src = image_src.to(device).half()

    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
    x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    # send to cuda
    x0_src = x0_src.to(device)

    negative_prompt =  "" # optionally add support for negative prompts (SD3)

    if model_type == 'SD3':
        x0_tar = FlowEditSD3(pipe,
                            scheduler,
                            x0_src,
                            src_prompt,
                            tar_prompt,
                            negative_prompt,
                            T_steps,
                            n_avg,
                            src_guidance_scale,
                            tar_guidance_scale,
                            n_min,
                            n_max,)
        
    elif model_type == 'FLUX':
        x0_tar = FlowEditFLUX(pipe,
                            scheduler,
                            x0_src,
                            src_prompt,
                            tar_prompt,
                            negative_prompt,
                            T_steps,
                            n_avg,
                            src_guidance_scale,
                            tar_guidance_scale,
                            n_min,
                            n_max,)
    else:
        raise NotImplementedError(f"Sampler type {model_type} not implemented")


    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
    image_tar = pipe.image_processor.postprocess(image_tar)


    return image_tar[0]


# title = "FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models"

intro = """
<h1 style="font-weight: 1000; text-align: center; margin: 0px;">FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models</h1>
<h3 style="margin-bottom: 10px; text-align: center;">
    <a href="https://arxiv.org/abs/2412.08629">[Paper]</a>&nbsp;|&nbsp;
    <a href="https://matankleiner.github.io/flowedit/">[Project Page]</a>&nbsp;|&nbsp;
    <a href="https://github.com/fallenshock/FlowEdit">[Code]</a>
</h3>
🚧 Early Version 🚧 Gradio demo for FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models. See our project page for more details.

<br>
<br>Edit your image using Flow models! upload an image, add a description of it, and specify the edits you want to make.
<h3>Notes:</h3>

<ol>
  <li>We use FLUX.1 dev and SD3 for the demo. The models are large and may take a while to load.</li>
  <li>We recommend 1024x1024 images for the best results. If the input images are too large, there may be out-of-memory errors.</li>
  <li>Default hyperparameters for each model used in the paper are provided as examples. Feel free to experiment with them as well.</li>
</ol>  

"""

# article = """
# 📝 **Citation**
# ```bibtex
# @article{aaa,
#   author    = {},
#   title     = {},
#   journal   = {},
#   year      = {2024},
#   url       = {}
# }
# ```
# """


with gr.Blocks() as demo:
    

    gr.HTML(intro)
    
    # with gr.Row():
    #     gr.LoginButton(value="Login to HF (For SD3 and FLUX access)", variant="primary")

    with gr.Row(equal_height=True):
        image_src = gr.Image(type="filepath", label="Source Image", value="inputs/cat.png",)
        image_tar = gr.Image(label="Output", type="pil", show_label=True, format="png",),

    with gr.Row():
        src_prompt = gr.Textbox(lines=2, label="Source Prompt", value="a cat sitting in the grass")

    with gr.Row():
        tar_prompt = gr.Textbox(lines=2, label="Target Prompt", value="a puppy sitting in the grass")

    with gr.Row():
        model_type = gr.Dropdown(["SD3", "FLUX"], label="Model Type", value="SD3")
        T_steps = gr.Number(value=50, label="Total Steps", minimum=1, maximum=50)
        n_max = gr.Number(value=33, label="n_max (control the strength of the edit)")

    with gr.Row():
        src_guidance_scale = gr.Slider(minimum=1.0, maximum=30.0, value=3.5, label="src_guidance_scale")
        tar_guidance_scale = gr.Slider(minimum=1.0, maximum=30.0, value=13.5, label="tar_guidance_scale")
    
    with gr.Row():
        submit_button = gr.Button("Run FlowEdit", variant="primary")


    with gr.Accordion(label="Advanced Settings", open=False):
        # additional inputs
        n_min = gr.Number(value=0, label="n_min (for improved style edits)")
        n_avg = gr.Number(value=1, label="n_avg (improve structure at the cost of runtime)", minimum=1)
        seed = gr.Number(value=42, label="seed")




    submit_button.click(
                        fn=FlowEditRun, 
                        inputs=[
                        image_src,
                        model_type,
                        T_steps,
                        src_guidance_scale,
                        tar_guidance_scale,
                        n_max,
                        src_prompt,
                        tar_prompt,
                        n_min,
                        n_avg,
                        seed,
                        ],
                        outputs=[
                        image_tar[0],
                        ],
                        )
    

    gr.Examples(
        label="Examples",
        examples=get_examples(),
        inputs=[image_src, model_type, T_steps, src_guidance_scale, tar_guidance_scale, n_max, src_prompt, tar_prompt, n_min, n_avg, seed, image_tar[0]],
        outputs=[image_tar[0]],
    )

    model_type.input(fn=on_model_change, inputs=[model_type], outputs=[T_steps, src_guidance_scale, tar_guidance_scale, n_max])


    # gr.HTML(article)
demo.queue()
demo.launch( )

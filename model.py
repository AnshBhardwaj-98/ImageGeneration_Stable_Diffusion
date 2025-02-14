
import matplotlib.pyplot as plt


from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "A majestic dragon with glowing emerald scales soars over a misty castle, its wings casting shadows on the enchanted valley below. Fiery breath lights up the stormy sky as embers swirl around. Epic, cinematic, ultra-detailed, 8K, fantasy realism."

images = pipe(prompt=prompt).images[0]

images.save('./result.jpg')



# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe.to("cuda")

# # if using torch < 2.0
# # pipe.enable_xformers_memory_efficient_attention()


# images = pipe(prompt=prompt).images[0]
# image.save("./result2.jpg")

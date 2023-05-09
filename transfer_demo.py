import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time
import io 
import os
import gdown

def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


class StyleTransfer:
    bin_path = "/Users/saket/Documents/workdir/harvard_courses/sem2/ac109b/harvard_CS109B/StyleTransfer/experiments"
    decoder_path = f"{bin_path}/decoder_iter_160000.pth"
    Trans_path = f"{bin_path}/transformer_iter_160000.pth"
    embedding_path = f"{bin_path}/embedding_iter_160000.pth"
    a = 1.0
    vgg = f"{bin_path}/vgg_normalised.pth"


paths = {
    StyleTransfer.vgg: "https://drive.google.com/uc?id=1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M",
    StyleTransfer.embedding_path: "https://drive.google.com/uc?id=1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B",
    StyleTransfer.decoder_path: "https://drive.google.com/uc?id=1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9",
    StyleTransfer.Trans_path: "https://drive.google.com/uc?id=1dnobsaLeE889T_LncCkAA2RkqzwsfHYy"
}

for local_path, url in paths.items():
    if not os.path.exists(local_path):
        gdown.download(url, local_path, quiet=False)
    else:
        print(f"File '{local_path}' already exists. Skipping download.")


def tensor_to_image_bytes(tensor):
    # Make sure the tensor is in CPU memory and has the correct shape
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    tensor = tensor.detach().numpy()

    # Convert the tensor values to the range [0, 255]
    tensor = (tensor * 255).clip(0, 255).astype("uint8")

    # Convert the tensor to a PIL Image
    image = Image.fromarray(tensor.transpose(1, 2, 0))

    # Save the PIL Image as bytes using BytesIO
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    return image_bytes


default_args = StyleTransfer()


def transfer_style(
        content_image,
        style_image,
        args = default_args
        ):
    """ Returns style transfered image """
    # Advanced options
    content_size=512
    style_size=512
    crop='store_true'
    save_ext='.jpg'
    preserve_color='store_true'
    alpha=args.a
    # content_paths = [Path(args.content)]
    # style_paths = [Path(args.style)]    


    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(args.decoder_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.Trans_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.embedding_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTR.StyTrans(vgg,decoder,embedding,Trans,args)
    network.eval()
    network.to(device)



    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    content_tf1 = content_transform()       
    content = content_tf(content_image.convert("RGB"))

    h,w,c=np.shape(content)    
    style_tf1 = style_transform(h,w)
    style = style_tf(style_image.convert("RGB"))


    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
        
    with torch.no_grad():
        output= network(content,style)       
    # output = output.cpu()

    return tensor_to_image_bytes(output[0])
   

if __name__ == "__main__":
    content_img = "/Users/saket/Documents/workdir/harvard_courses/sem2/ac109b/harvard_CS109B/final_project/stytr_2/examples/content/content_feynman_1.webp"
    style_img = "/Users/saket/Documents/workdir/harvard_courses/sem2/ac109b/harvard_CS109B/final_project/stytr_2/examples/style/style_vangogh_1.webp"

    content_img = Image.open(content_img)
    style_img = Image.open(style_img)

    output = transfer_style(content_img, style_img)
    print(output)
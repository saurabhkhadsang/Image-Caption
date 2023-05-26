from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
import requests
import torch
import numpy as np
from PIL import Image
import pickle
# from torchvision import transforms
# from datasets import load_dataset
# import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2

import warnings
warnings.filterwarnings('ignore')


model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer       = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def show_n_generate(url, greedy = True, model = model_raw):
    image = Image.open(requests.get(url, stream =True).raw)
    pixel_values   = image_processor(image, return_tensors ="pt").pixel_values
    # cv2.imshow("window_name", image)
    # plt.imshow(np.asarray(image))
    # plt.show()

    if greedy:
        generated_ids  = model.generate(pixel_values, max_new_tokens = 30)
    else:
        generated_ids  = model.generate(
            pixel_values,
            do_sample=True,
            max_new_tokens = 30,
            top_k=5)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


    # response = requests.get(url, stream=True)
    # image = Image.open(response.raw)

    # # Convert the PIL image to a numpy array
    # image_np = np.array(image)

    # # Display the image using OpenCV
    # cv2.imshow("Image", image_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


url = "https://www.nrdc.org/sites/default/files/styles/social_sharing_1200x630/public/media-uploads/wlds43_654640_2400.jpg?h=c3635fa2&itok=S2-v9ebR"

show_n_generate(url, greedy = False)
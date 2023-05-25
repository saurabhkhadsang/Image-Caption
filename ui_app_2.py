import streamlit as st
import torch
from PIL import Image
import requests
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

# Load the pre-trained models
model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def get_model():
    return model_raw

def get_image_processor():
    return image_processor

def get_tokenizer():
    return tokenizer

# Function to generate captions for the input image URL
def generate_captions(url, greedy=True):
    model = get_model()
    image_processor = get_image_processor()
    tokenizer = get_tokenizer()

    # Load and process the image
    image = Image.open(requests.get(url, stream=True).raw)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    if greedy:
        generated_ids = model.generate(pixel_values, max_new_tokens=30)
    else:
        generated_ids = model.generate(
            pixel_values,
            do_sample=True,
            max_new_tokens=30,
            top_k=5
        )

    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Streamlit app
def main():
    st.title("Image Captioning")

    # Input URL
    url = st.text_input("Enter the image URL")

    # Generate captions
    if st.button("Generate Captions"):
        if url:
            with st.spinner("Generating captions..."):
                captions = generate_captions(url, greedy=False)
            st.success("Captions generated successfully!")
            st.text(captions)

            # Display the image
            image = Image.open(requests.get(url, stream=True).raw)
            st.image(image, caption='Input Image', use_column_width=True)
        else:
            st.warning("Please enter an image URL.")

if __name__ == "__main__":
    main()

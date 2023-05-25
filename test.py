import streamlit as st
import urllib
import os


def save_image(image, path):
    with open(path, "wb") as f:
        f.write(image.getbuffer())


def main():
    st.title("Image Saver")

    # Input selection: URL or local file
    st.sidebar.title("Input Options")
    input_option = st.sidebar.radio("Select input type:", ("URL", "Local File"))

    if input_option == "URL":
        image_url = st.sidebar.text_input("Enter the image URL:")
        if image_url:
            try:
                image = urllib.request.urlopen(image_url).read()
                st.image(image, caption="Input Image")
            except Exception as e:
                st.error(f"Error loading image: {e}")

    else:  # input_option == "Local File"
        image_file = st.sidebar.file_uploader("Upload an image file:")
        if image_file is not None:
            try:
                image = image_file.read()
                st.image(image, caption="Input Image")
            except Exception as e:
                st.error(f"Error loading image: {e}")

    # Save button
    if st.button("Save Image"):
        if "image" in locals():
            save_path = st.text_input("Enter the save path:", "output.jpg")
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            try:
                save_image(image, save_path)
                st.success("Image saved successfully!")
            except Exception as e:
                st.error(f"Error saving image: {e}")

    # Output image
    if "image" in locals():
        st.subheader("Output Image")
        st.image(image, caption="Output Image")


if __name__ == "__main__":
    main()

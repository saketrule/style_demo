import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tempfile
import io

# Assuming a trained model for style transfer is available
import sys
# sys.path.append("/Users/saket/Documents/workdir/harvard_courses/sem2/ac109b/harvard_CS109B/StyleTransfer")
from transfer_demo import transfer_style

st.set_page_config(page_title="Style Transfer", layout="centered", initial_sidebar_state="collapsed")

st.title("Image Style Transfer")
st.markdown("Upload a content image and a style image to create a stylized output image.")

content_file = st.file_uploader("Choose a Content Image", type=["jpg", "jpeg", "png", "webp"])
style_file = st.file_uploader("Choose a Style Image", type=["jpg", "jpeg", "png", "webp"])

if content_file is not None and style_file is not None:
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)

    st.header("Content Image")
    st.image(content_image, use_column_width=True)

    st.header("Style Image")
    st.image(style_image, use_column_width=True)

    st.header("Output Image")
    with st.spinner("Performing style transfer..."):
        output_image = transfer_style(content_image, style_image)
        image_file = io.BytesIO(output_image)
        st.image(image_file, use_column_width=True)

    if st.button("Download Output Image"):
        buffered = BytesIO()
        output_image.save(buffered, format="JPEG")
        img_data = buffered.getvalue()

        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(img_data)
            st.markdown(f"<a href='file://{fp.name}' download='stylized_output.jpg'>Click here to download the output image</a>", unsafe_allow_html=True)

else:
    st.markdown("**Please upload both content and style images to see the stylized output.**")

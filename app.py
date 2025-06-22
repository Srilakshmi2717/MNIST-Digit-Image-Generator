import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import Generator  # Import your trained generator class
import numpy as np

# Load trained Generator model
generator = Generator()
generator.load_state_dict(torch.load("generator.pth", map_location="cpu"))
generator.eval()

# Function to generate 5 distinct images for the chosen digit
def generate_images(digit, num_images=5):
    z = torch.randn(num_images, 100)
    labels = torch.full((num_images,), digit, dtype=torch.long)
    with torch.no_grad():
        generated_images = generator(z, labels)
    return generated_images

# Streamlit UI
st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("Handwritten Digit Image Generator")
st.subheader("Generate synthetic MNIST-like Images using your trained model.")

# Dropdown input
digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate Images"):
    st.write(f"Generated images of digit {digit}")
    images = generate_images(digit)

    # Display 5 images in a row
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        img = (images[i].squeeze().numpy() + 1) / 2
        ax.imshow(img, cmap="gray")
        ax.axis("off")

    st.pyplot(fig)
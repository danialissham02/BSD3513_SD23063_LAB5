#CPU web-based image classification app using Streamlit and PyTorch
# Step 1: Create Streamlit app and configure page
import streamlit as st
st.set_page_config(
    page_title="CPU-Based Image Classification",
    layout="centered"
)

# Step 2: Import required libraries
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import torch.nn.functional as F

st.title("Image Classification using ResNet18 (CPU)")
st.write("Upload an image to classify it using a pre-trained ResNet18 model.")

# Step 3: Configure CPU-only execution
device = torch.device("cpu")

# Step 4: Load pre-trained ResNet18 model
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.to(device)
model.eval()

# Step 5: Image preprocessing transformations
preprocess = weights.transforms()

# Load ImageNet class labels
labels = weights.meta["categories"]

# Step 6: Image upload interface
uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 7: Convert image to tensor & inference
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # Step 8: Softmax & top-5 predictions
    probabilities = F.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    results = {
        "Class": [labels[i] for i in top5_idx],
        "Probability": top5_prob.cpu().numpy()
    }

    df = pd.DataFrame(results)

    st.subheader("Top-5 Predictions")
    st.table(df)

    # Step 9: Bar chart visualization
    st.subheader("Prediction Probabilities")
    st.bar_chart(df.set_index("Class"))

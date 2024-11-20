import torch
import clip
from PIL import Image
import numpy as np
import os
from labels_to_descriptions import LABELS_TO_DESCRIPTIONS

# Initialize the CLIP model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

def encode_image(image_path):
    """
    Encodes the image using the CLIP model.
    """
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features

def encode_text(texts):
    """
    Encodes a list of texts using the CLIP model.
    """
    text_input = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    return text_features

def match_image_to_description(image_path, descriptions):
    """
    Matches the image with the best description using CLIP model.
    """
    # Encode image
    image_features = encode_image(image_path)
    
    # Encode text descriptions
    text_features = encode_text(descriptions)

    # Calculate similarity scores
    similarity = torch.cosine_similarity(image_features, text_features)

    # Get the best matching description
    best_match_idx = similarity.argmax().item()
    best_description = descriptions[best_match_idx]
    score = similarity[best_match_idx].item()

    return best_description, score

def zero_shot_classification(image_path, descriptions):
    """
    Performs zero-shot classification by matching the image with descriptions.
    """
    best_description, score = match_image_to_description(image_path, descriptions)
    print(f"[INFO] Image: {image_path} - Best Description: '{best_description}' with score: {score}")
    return best_description, score

if __name__ == "__main__":
    # Example of how to run the zero-shot classification on a single image
    image_path = r"D:\CLIP\Dataset\images\Abyssinian_1.jpg"  # Example image
    descriptions = list(LABELS_TO_DESCRIPTIONS.values())  # List of all descriptions

    zero_shot_classification(image_path, descriptions)

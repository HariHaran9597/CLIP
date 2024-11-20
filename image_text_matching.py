import torch
import clip
from torch.utils.data import DataLoader
from data_loader import OxfordPetsDataset
from labels_to_descriptions import LABELS_TO_DESCRIPTIONS

# Initialize the CLIP model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

def encode_image(image):
    """
    Encodes the image using the CLIP model.
    """
    image_input = image.unsqueeze(0).to(device)  # Image is already preprocessed to tensor format
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features

def encode_text(texts):
    """
    Encodes a list of texts (breed descriptions) using the CLIP model.
    """
    text_input = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    return text_features

def match_image_to_description(image, descriptions):
    """
    Matches the image with the best description using CLIP model.
    """
    # Encode image
    image_features = encode_image(image)
    
    # Encode text descriptions (breed descriptions)
    text_features = encode_text(descriptions)

    # Calculate similarity scores
    similarity = torch.cosine_similarity(image_features, text_features)

    # Get the best matching description
    best_match_idx = similarity.argmax().item()
    best_description = descriptions[best_match_idx]
    score = similarity[best_match_idx].item()

    return best_description, score

def evaluate_classification(dataloader):
    """
    Evaluate zero-shot classification across a dataset of images.
    """
    correct = 0
    total = 0
    for images, descriptions, labels in dataloader:
        images = images.to(device)  # Move images to the device
        for idx, image in enumerate(images):
            description = descriptions[idx]
            best_description, score = match_image_to_description(image, descriptions)
            print(f"[INFO] Image: {labels[idx]} - Best Description: '{best_description}' with score: {score}")
            
            # Assume the best description match is the ground truth class
            if best_description in LABELS_TO_DESCRIPTIONS.values():
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"[RESULT] Accuracy: {accuracy:.4f}")

# Usage Example
if __name__ == "__main__":
    image_folder = r"D:\CLIP\Dataset\images"  # Replace with your image folder path
    dataset = OxfordPetsDataset(image_folder, LABELS_TO_DESCRIPTIONS, preprocess)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    evaluate_classification(dataloader)

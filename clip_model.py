import clip
import torch

# Load CLIP model and preprocess pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_clip = clip.load('ViT-B/32', device=device)
print("[INFO] CLIP model loaded successfully")

# Example of encoding an image and a text
def encode_image(image):
    image_input = preprocess_clip(image).unsqueeze(0).to(device)  # Apply preprocess to PIL image
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    print("[INFO] Image encoded successfully")
    return image_features

def encode_text(text):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    print("[INFO] Text encoded successfully")
    return text_features

# Test encoding functions
if __name__ == "__main__":
    # Sample test cases
    sample_text = "A cat with a fluffy tail sitting on a couch."
    encode_text(sample_text)

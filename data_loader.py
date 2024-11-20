import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from labels_to_descriptions import LABELS_TO_DESCRIPTIONS
import clip

# Preprocessing for CLIP model
preprocess = clip.load("ViT-B/32", device="cpu")[1]  # Use preprocess from CLIP model

class OxfordPetsDataset(Dataset):
    """
    Custom Dataset for Oxford Pets dataset with image-text matching.
    """
    def __init__(self, image_folder, labels_to_descriptions, preprocess):
        self.image_folder = image_folder
        self.labels_to_descriptions = labels_to_descriptions
        self.preprocess = preprocess
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_name = os.path.basename(image_path).split('_')[0]  # Extract the label from the file name (e.g., 'Abyssinian')
        description = self.labels_to_descriptions.get(label_name, "")  # Get corresponding description
        
        # Load and preprocess the image
        image = Image.open(image_path)
        image = self.preprocess(image)  # Preprocess image for CLIP
        
        return image, description, label_name

# Usage Example
if __name__ == "__main__":
    image_folder = r"D:\CLIP\Dataset\images"  # Your images folder path
    dataset = OxfordPetsDataset(image_folder, LABELS_TO_DESCRIPTIONS, preprocess)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Loaded {len(dataset)} images")

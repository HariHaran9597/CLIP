import os
from zero_shot_classification import zero_shot_classification

# Function to evaluate model on the Oxford Pets dataset
def evaluate_model(image_folder, annotation_folder, candidate_descriptions):
    total_similarity = 0
    count = 0
    
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, filename)
            annotation_path = os.path.join(annotation_folder, filename.replace('.jpg', '.xml'))
            
            best_description, similarity_score = zero_shot_classification(image_path, candidate_descriptions)
            total_similarity += similarity_score
            count += 1
            print(f"[INFO] Processed {filename} with best description: '{best_description}' (Score: {similarity_score})")
    
    average_similarity = total_similarity / count
    print(f"[RESULT] Average Similarity across dataset: {average_similarity}")

# Example evaluation
if __name__ == "__main__":
    candidate_descriptions = [
        "A fluffy cat sitting on a sofa near a window.",
        "A dog with a brown coat running through grass.",
        "A kitten playing with a ball of yarn."
    ]
    evaluate_model(r'D:\CLIP\Dataset\images', r'D:\CLIP\Dataset\annotations\xmls', candidate_descriptions)

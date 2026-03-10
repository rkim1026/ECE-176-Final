import os
import random  # <-- Added the random module!
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

# --- 1. Our Custom, No-Nonsense Image Loader ---
class FlatFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        
        # Look for these file types
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        # Grab every file in the folder that ends with a valid extension
        self.image_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(valid_extensions)
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.folder_path, filename)
        
        # Open image and force it to RGB (removes alpha channels that cause crashes)
        image = Image.open(img_path).convert('RGB') 
        
        if self.transform:
            image = self.transform(image)
            
        return image, filename

# --- 2. The Random Masking Logic ---
def apply_random_mask(img, img_size=128, hole_size=64):
    # Calculate the maximum starting index so the hole doesn't bleed off the edge
    max_idx = img_size - hole_size
    
    # Pick a random top-left corner for the mask
    start_y = random.randint(0, max_idx)
    start_x = random.randint(0, max_idx)
    
    masked_img = img.clone()
    
    # Apply the mask (zeroing out the pixels) at the random location
    masked_img[:, start_y:start_y+hole_size, start_x:start_x+hole_size] = 0 
    
    # Extract the exact patch that was just erased
    gt_patch = img[:, start_y:start_y+hole_size, start_x:start_x+hole_size] 
    
    return masked_img, gt_patch

# --- 3. The Processing Loop ---
def process_and_save_dataset(input_dir, output_dir, img_size=128, hole_size=64):
    # Output folders (kept 'class_0' just in case you use ImageFolder later)
    masked_dir = os.path.join(output_dir, 'masked_images', 'class_0')
    gt_dir = os.path.join(output_dir, 'ground_truth_patches', 'class_0')
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
    
    # Use our new custom dataset instead of ImageFolder!
    dataset = FlatFolderDataset(input_dir, transform=transform)
    print(f"Found {len(dataset)} images in {input_dir}. Processing...")
    
    for idx in range(len(dataset)):
        img, filename = dataset[idx] # Our custom dataset returns the filename directly
        
        # --- Updated to use the new random mask function ---
        masked_img, gt_patch = apply_random_mask(img, img_size, hole_size)
        
        save_image(masked_img, os.path.join(masked_dir, filename))
        save_image(gt_patch, os.path.join(gt_dir, filename))
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} images...")
            
    print(f"Done! Saved to: {output_dir}")

if __name__ == "__main__":
    # Put all your images directly inside this folder, no subfolders needed!
    INPUT_FOLDER = './fruit_data' 
    OUTPUT_FOLDER = './processed_fruit_data'
    
    process_and_save_dataset(INPUT_FOLDER, OUTPUT_FOLDER)
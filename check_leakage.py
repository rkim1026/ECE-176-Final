import os
import hashlib

def get_file_hash(filepath):
    """Returns the MD5 hash of a file."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Set the paths to your directories
dir_all = 'plant_dataset_all'
dir_test = 'plant_dataset_final_test'

print("Hashing files in plant_dataset_all...")
all_hashes = {}
for root, _, files in os.walk(dir_all):
    for file in files:
        filepath = os.path.join(root, file)
        all_hashes[get_file_hash(filepath)] = filepath

print("Checking for duplicates in plant_dataset_final_test...")
duplicates = []
for root, _, files in os.walk(dir_test):
    for file in files:
        filepath = os.path.join(root, file)
        file_hash = get_file_hash(filepath)
        if file_hash in all_hashes:
            duplicates.append((filepath, all_hashes[file_hash]))

if duplicates:
    print(f"\nFound {len(duplicates)} duplicate image(s)!")
    for test_file, all_file in duplicates:
        print(f"\nIn Test: {test_file}")
        print(f"In All:  {all_file}")
        
        # Uncomment the next line to automatically delete the duplicate from plant_dataset_all
        # os.remove(all_file) 
        # print(f"Deleted from plant_dataset_all: {all_file}")
else:
    print("\nNo duplicates found. Your test set is entirely isolated!")
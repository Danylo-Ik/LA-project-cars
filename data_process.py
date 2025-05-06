import pickle
import os
import cv2 as cv
import numpy as np
from decomposition import svd


def save_dataset(dataset, filename="chars74k_vectors_bases.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {filename}")


def load_dataset(filename="chars74k_vectors_bases.pkl"):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Dataset loaded from {filename}")
    return dataset


def get_label(folder_name):
    try:
        index = int(folder_name[-3:])
        if 1 <= index <= 10:
            return str(index - 1)
        elif 11 <= index <= 36:
            return chr(ord('A') + index - 11)
        else:
            return None
    except ValueError:
        return None


def load_and_flatten(dataset_path, image_size=(28, 28), max_per_class=1000):
    dataset = {}
    
    for folder_name in sorted(os.listdir(dataset_path)):
        # label = get_label(folder_name)
        label = str(folder_name)
        if label is None:
            continue

        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
        # image_files = image_files[:max_per_class]

        print(f"Processing folder: {folder_name} with label: {label}, number of images: {len(image_files)}")
        flattened = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv.resize(img, image_size)
            img_bin = (img_resized < 128).astype(np.uint8)
            vector = img_bin.flatten().astype(np.float32)
            flattened.append(vector)
        
        if flattened:
            dataset[label] = flattened

    return dataset


def save_svd_dataset(original_dataset, filename="chars74k_vectors_bases.pkl", k=50):
    compressed_dataset = {}

    for label, images in original_dataset.items():
        print(f"Creating basis for label: {label}, number of images: {len(images)}")
        
        A = np.stack([img.flatten().astype(np.float32) for img in images], axis=1)
        mean = A.mean(axis=1, keepdims=True)
        A_centered = A - mean
        U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
        U_k = U[:, :k]

        compressed_dataset[label] = {
            'U': U_k,
            'mean': mean.flatten()
        }

    with open(filename, "wb") as f:
        pickle.dump(compressed_dataset, f)


if __name__ == "__main__":
    base_path = "/Users/danyilikonnikov/Downloads/CNN letter Dataset"
    dataset = load_and_flatten(base_path)
    save_svd_dataset(dataset, filename="chars74k_vectors_bases.pkl", k=60)

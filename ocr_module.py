import numpy as np
import cv2 as cv
from collections import deque
from decomposition import svd

def binarize(image, threshold=128):
    return (image < threshold).astype(np.uint8)

def connected_components(binary_image):
    rows, cols = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)
    components = []
    
    # 8-connectivity BFS
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1 and not visited[i, j]:
                component = []
                queue = deque([(i, j)])
                
                while queue:
                    x, y = queue.popleft()
                    if visited[x, y]:
                        continue
                    
                    visited[x, y] = True
                    component.append((x, y))
                    
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and binary_image[nx, ny] == 1 and not visited[nx, ny]:
                                queue.append((nx, ny))
                
                components.append(component)
    
    return components

def extract_characters(binary_image, components, padding=2, target_size=(28, 28), min_area=2000, max_area=5000):
    characters = []

    for comp in components:
        if len(comp) < min_area or len(comp) > max_area:
            continue  # skip small noise

        ys = [i for i, j in comp]
        xs = [j for i, j in comp]

        top = max(min(ys) - padding, 0)
        bottom = min(max(ys) + padding + 1, binary_image.shape[0])
        left = max(min(xs) - padding, 0)
        right = min(max(xs) + padding + 1, binary_image.shape[1])

        char_crop = binary_image[top:bottom, left:right]
        resized_char = cv.resize(char_crop, target_size, interpolation=cv.INTER_NEAREST)

        x_center = (left + right) // 2
        characters.append((x_center, resized_char))

    characters.sort(key=lambda tup: tup[0])

    return [img for _, img in characters]


def recognize_character(image, dataset):
    image_vec = image.flatten().astype(np.float32)

    best_match = None
    best_error = float('inf')

    for label, data in dataset.items():
        U = data['U']
        mean = data['mean']

        centered = image_vec - mean

        projection = U @ (U.T @ centered)
        error = np.linalg.norm(centered - projection)

        if error < best_error:
            best_error = error
            best_match = label

    return best_match

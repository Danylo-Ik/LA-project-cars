import numpy as np


def svd(A, max_iter=50, tol=1e-6):
    m, n = A.shape
    k = min(m, n)

    U = np.zeros((m, k))
    Sigma = np.zeros(k)
    Vt = np.zeros((k, n))

    ATA = A.T @ A

    for i in range(k):
        v = np.random.rand(n)
        v /= np.linalg.norm(v)

        for _ in range(max_iter):
            v_new = ATA @ v
            for j in range(i):
                v_new -= (v_new @ Vt[j]) * Vt[j]

            v_new_norm = np.linalg.norm(v_new)
            if v_new_norm < tol:
                break

            v = v_new / v_new_norm

        Av = A @ v
        sigma = np.linalg.norm(Av)

        if sigma > tol:
            u = Av / sigma
        else:
            u = np.zeros(m)

        U[:, i] = u
        Sigma[i] = sigma
        Vt[i, :] = v

        ATA -= sigma**2 * np.outer(v, v)

    return U, Sigma, Vt


def denoise_image(image, k=10):
    img_float = image.astype(np.float32) / 255.0

    U, Sigma, Vt = svd(img_float)

    U_k = U[:, :k]
    Sigma_k = Sigma[:k]
    Vt_k = Vt[:k, :]

    denoised = U_k @ np.diag(Sigma_k) @ Vt_k
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

    return denoised

# if __name__ == "__main__":
#     import cv2
#     image = cv2.imread('test images/Car with Plates.jpg', cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError("Image not found or unable to load.")
#     # Denoise the image
#     denoised_image = denoise_image(image, 200)

#     cv2.imshow('Original Image', image)
#     cv2.imshow('Denoised Image', denoised_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

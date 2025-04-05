    k = int(min(binary.shape) * 0.25)
    print(f"Using level {k} for SVD denoising")
    binary = denoise_image(binary, k)
    cv.imwrite("denoised.jpg", binary)
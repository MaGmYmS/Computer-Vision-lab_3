import time

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm


def generate_sobel_filter(size):
    if size % 2 == 0 or size < 3:
        raise ValueError("Size must be an odd number greater than or equal to 3")
    sobel_x, sobel_y = [], []

    match size:
        case 3:
            # sobel_x = np.array([[-1, 0, 1],
            #                     [-2, 0, 2],
            #                     [-1, 0, 1]])
            #
            # sobel_y = np.array([[-1, -2, -1],
            #                     [0, 0, 0],
            #                     [1, 2, 1]])

            sobel_x = np.array([[-1 / 2, 0, 1 / 2],
                                [-1, 0, 1],
                                [-1 / 2, 0, 1 / 2]])

            sobel_y = np.array([[-1 / 2, -1, -1 / 2],
                                [0, 0, 0],
                                [1 / 2, 1, 1 / 2]])
        case 5:
            sobel_x = np.array([[-2 / 8, -1 / 5, 0, 1 / 5, 2 / 8],
                                [-2 / 5, -1 / 2, 0, 1 / 2, 2 / 5],
                                [-2 / 4, -1 / 1, 0, 1 / 1, 2 / 4],
                                [-2 / 5, -1 / 2, 0, 1 / 2, 2 / 5],
                                [-2 / 8, -1 / 5, 0, 1 / 5, 2 / 8]])

            sobel_y = np.array([[-2 / 8, -2 / 5, -2 / 4, -2 / 5, -2 / 8],
                                [-1 / 5, -1 / 2, -1 / 1, -1 / 2, -1 / 5],
                                [0, 0, 0, 0, 0],
                                [1 / 5, 1 / 2, 1 / 1, 1 / 2, 1 / 5],
                                [2 / 8, 2 / 5, 2 / 4, 2 / 5, 2 / 8]])
        case 7:
            sobel_x = np.array([[-3 / 18, -2 / 13, -1 / 10, 0, 1 / 10, 2 / 13, 3 / 18],
                                [-3 / 13, -2 / 8, -1 / 5, 0, 1 / 5, 2 / 8, 3 / 13],
                                [-3 / 10, -2 / 5, -1 / 2, 0, 1 / 2, 2 / 5, 3 / 10],
                                [-3 / 9, -2 / 4, -1 / 1, 0, 1 / 1, 2 / 4, 3 / 9],
                                [-3 / 10, -2 / 5, -1 / 2, 0, 1 / 2, 2 / 5, 3 / 10],
                                [-3 / 13, -2 / 8, -1 / 5, 0, 1 / 5, 2 / 8, 3 / 13],
                                [-3 / 18, -2 / 13, -1 / 10, 0, 1 / 10, 2 / 13, 3 / 18]])

            sobel_y = np.array([[-3 / 18, -3 / 13, -3 / 10, -3 / 9, -3 / 10, -3 / 13, -3 / 18],
                                [-2 / 13, -2 / 8, -2 / 5, -2 / 4, -2 / 5, -2 / 8, -2 / 13],
                                [-1 / 10, -1 / 5, -1 / 2, -1 / 1, 0, -1 / 5, -1 / 10],
                                [0, 0, 0, 0, 0, 0, 0],
                                [1 / 10, 1 / 5, 1 / 2, 1 / 1, 0, 1 / 5, 1 / 10],
                                [2 / 13, 2 / 8, 2 / 5, 2 / 4, 2 / 5, 2 / 8, 2 / 13],
                                [3 / 18, 3 / 13, 3 / 10, 3 / 9, 3 / 10, 3 / 13, 3 / 18]])

    return sobel_x, sobel_y


def apply_custom_sobel_filter(image, filter_size):
    grayscale_image = image.convert("L")
    width, height = grayscale_image.size

    sobel_x, sobel_y = generate_sobel_filter(filter_size)
    half_size = filter_size // 2

    new_image = Image.new("L", (width, height))

    for x in tqdm(range(half_size, width - half_size)):
        for y in range(half_size, height - half_size):
            pixel_matrix = np.array([[grayscale_image.getpixel((x + i - half_size, y + j - half_size))
                                      for j in range(filter_size)] for i in range(filter_size)])

            gradient_x = np.sum(pixel_matrix * sobel_x)
            gradient_y = np.sum(pixel_matrix * sobel_y)

            gradient_magnitude = int(np.sqrt(np.power(gradient_x, 2) + np.power(gradient_y, 2)))
            new_image.putpixel((x, y), gradient_magnitude)

    return new_image


def difference_of_gaussian(image, sigma1, sigma2):
    blurred1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    blurred2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    dog = blurred1 - blurred2
    return dog


def compare_result(image, method):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, filter_size in enumerate([3, 5, 7]):
        start_time = time.time()
        output_image = method(image, filter_size)
        end_time = time.time()
        execution_time = end_time - start_time
        output_image_path = f"images/output_image_{method.__name__}_{filter_size}.jpg"
        output_image.save(output_image_path)
        ax = axes[idx]
        ax.imshow(output_image, cmap='gray')
        ax.set_title(f"Filter Size: {filter_size}\nExecution Time: {execution_time:.4f} sec")
        ax.axis('off')
    plt.tight_layout()
    fig.canvas.manager.set_window_title(method.__name__)
    plt.show()


if __name__ == "__main__":
    input_image_path = "images/flower.jpg"
    input_image = Image.open(input_image_path)
    cv2.imshow("difference of gaussian", difference_of_gaussian(image=input_image, sigma1=2, sigma2=1))
    # methods = [apply_custom_sobel_filter]
    # compare_result(input_image, apply_custom_sobel_filter)


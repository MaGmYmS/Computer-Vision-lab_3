import math
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


# def get_laplacian_of_gaussian_filter(kernel_size, sigma):
#     """
#     Создает ядро фильтра Гаусса заданного размера и сигмой во 2 производной.
#     """
#     kernel = [[0] * kernel_size for _ in range(kernel_size)]
#     center = kernel_size // 2
#
#     for x in range(kernel_size):
#         for y in range(kernel_size):
#             distance_sq = ((x - center) ** 2 + (y - center) ** 2 - 2 * sigma ** 2) / sigma ** 4
#             # kernel[x][y] = ((1 / (2 * math.pi * sigma ** 4)) * distance_sq
#             #                 * np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)))
#             kernel[x][y] = (distance_sq * np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)))
#
#     total = sum(sum(row) for row in kernel)
#     for x in range(kernel_size):
#         for y in range(kernel_size):
#             kernel[x][y] /= total
#
#     kernel[center][center] = - kernel[center][center]
#     for i in range(kernel_size):
#         print(*kernel[i])
#     return kernel


def custom_gaussian_kernel(kernel_size, sigma):
    """
    Создает ядро фильтра Гаусса заданного размера и сигмой.
    """
    kernel = np.zeros((kernel_size, kernel_size))

    center = kernel_size // 2
    for x in range(kernel_size):
        for y in range(kernel_size):
            distance_sq = (x - center) ** 2 + (y - center) ** 2
            kernel[x, y] = np.exp(-distance_sq / (2 * sigma ** 2))

    kernel /= np.sum(kernel)

    return kernel


def apply_custom_gaussian_filter(image, sigma):
    # Преобразуем изображение в массив numpy
    image_array = np.array(image)

    kernel_size = int(sigma * 6 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    half_kernel_size = kernel_size // 2

    # Добавляем отступы к изображению
    padded_image = np.pad(image_array,
                          ((half_kernel_size, half_kernel_size), (half_kernel_size, half_kernel_size), (0, 0)),
                          mode='edge')

    filtered_image = np.zeros_like(padded_image, dtype=float)

    # Получаем ядро фильтра Гаусса
    gaussian_kernel = custom_gaussian_kernel(kernel_size, sigma)

    total_iterations = 3 * (padded_image.shape[0] - half_kernel_size * 2) * (
            padded_image.shape[1] - half_kernel_size * 2)

    with tqdm(total=total_iterations, desc=f"Gaussian filter {sigma}") as pbar:
        for channel in range(3):
            for x in range(half_kernel_size, padded_image.shape[0] - half_kernel_size):
                for y in range(half_kernel_size, padded_image.shape[1] - half_kernel_size):
                    region = padded_image[x - half_kernel_size:x + half_kernel_size + 1,
                             y - half_kernel_size:y + half_kernel_size + 1, channel]

                    weighted_sum = np.sum(region * gaussian_kernel)

                    # Записываем в отфильтрованное изображение нормированное значение
                    filtered_image[x, y, channel] = weighted_sum

                    pbar.update(1)

    # Обрезаем отступы и приводим к RGB формату
    filtered_image_rgb = filtered_image[half_kernel_size:-half_kernel_size, half_kernel_size:-half_kernel_size]

    # Преобразуем массив numpy обратно в изображение PIL
    filtered_image_pil = Image.fromarray(filtered_image_rgb.astype(np.uint8))

    return filtered_image_pil


def apply_custom_laplacian_of_gaussian_filter(image, sigma):
    # Применяем фильтр Гаусса
    blurred_image = np.array(apply_custom_gaussian_filter(image, sigma))

    # Фильтр Лапласа
    laplacian_filter = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    half_kernel_size = 3 // 2

    # Создаем массив для отфильтрованного изображения
    filtered_image = np.zeros_like(np.array(image), dtype=float)
    total_iterations = 3 * (blurred_image.shape[0] - half_kernel_size * 2) * (
            blurred_image.shape[1] - half_kernel_size * 2)
    with tqdm(total=total_iterations, desc=f"Laplacian of Gaussian filter {sigma}") as pbar:
        for channel in range(3):
            for x in range(half_kernel_size, blurred_image.shape[0] - half_kernel_size):
                for y in range(half_kernel_size, blurred_image.shape[1] - half_kernel_size):
                    region = blurred_image[x - half_kernel_size:x + half_kernel_size + 1,
                             y - half_kernel_size:y + half_kernel_size + 1, channel]

                    weighted_sum = np.sum(region * laplacian_filter)

                    # Записываем в отфильтрованное изображение нормированное значение
                    filtered_image[x, y, channel] = weighted_sum
                    pbar.update(1)

    # Преобразуем изображение в оттенки серого
    gray_filtered_image = cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    log_filtered_image = Image.fromarray(gray_filtered_image.astype(np.uint8))

    return log_filtered_image


def apply_dog_filter(image, sigma1):
    """
    Применяет фильтр DoG (Difference of Gaussians) к изображению.
    """
    sigma2 = sigma1 * 10.485
    # Применяем Гауссово размытие с двумя ядрами Гаусса
    blurred_image1 = np.array(apply_custom_gaussian_filter(image, sigma1))
    blurred_image2 = np.array(apply_custom_gaussian_filter(image, sigma2))

    # Вычисляем разность двух размытых изображений
    dog_image = np.abs(blurred_image1 - blurred_image2)
    gray_dog_image = cv2.cvtColor(dog_image, cv2.COLOR_BGR2GRAY)

    # Преобразуем массив numpy обратно в изображение PIL
    dog_image_pil = Image.fromarray(gray_dog_image.astype(np.uint8))

    return dog_image_pil


def compare_result(image, methods, filter_sizes):
    fig, axes = plt.subplots(len(methods), len(filter_sizes), figsize=(15, 15))

    for method_idx, method in enumerate(methods):
        for row_idx in range(len(filter_sizes[method])):
            filter_size = filter_sizes[method][row_idx]
            start_time = time.time()
            output_image = method(image, filter_size)
            end_time = time.time()
            execution_time = end_time - start_time

            ax = axes[method_idx][row_idx]
            ax.imshow(output_image, cmap='gray')
            ax.set_title(f"{method.__name__}\nFilter Size: {filter_size}\nExecution Time: {execution_time:.4f} sec")
            ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # input_image_path = "images/image3_gauss_1.png"
    input_image_path = "images/flower.jpg"

    input_image = Image.open(input_image_path)
    out = apply_custom_laplacian_of_gaussian_filter(input_image, 2)
    out.show()
    out.save("123.png")
    # filter_sizes_dict = {apply_custom_sobel_filter: [3, 5, 7], apply_custom_laplacian_of_gaussian_filter: [0.5, 2, 5],
    #                      apply_dog_filter: [0.2, 0.5, 1]}
    # methods_filter = [apply_custom_sobel_filter, apply_custom_laplacian_of_gaussian_filter, apply_dog_filter]
    # compare_result(input_image, methods_filter, filter_sizes_dict)

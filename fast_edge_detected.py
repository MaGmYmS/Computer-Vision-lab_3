import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


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
    sobel_x, sobel_y = generate_sobel_filter(filter_size)

    gradient_x = convolve2d(image, sobel_x, mode='same', boundary='symm')
    gradient_y = convolve2d(image, sobel_y, mode='same', boundary='symm')

    # Вычисляем абсолютное значение градиента
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2).astype(np.uint8)

    return gradient_magnitude


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


def apply_custom_gaussian_filter(image_array, sigma):
    kernel_size = int(sigma * 6 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    gaussian_kernel = custom_gaussian_kernel(kernel_size, sigma)

    filtered_image = convolve2d(image_array, gaussian_kernel, mode='same', boundary='symm')

    return filtered_image.astype(np.uint8)


def apply_difference_of_gaussian_filter(image, sigma1):
    """
    Применяет фильтр DoG (Difference of Gaussians) к изображению.
    """
    sigma2 = sigma1 * 1.6

    # Преобразуем изображение в массив numpy
    image_array = np.array(image)

    blurred_image1 = apply_custom_gaussian_filter(image_array, sigma1)
    blurred_image2 = apply_custom_gaussian_filter(image_array, sigma2)

    # Вычисляем разность двух размытых изображений
    dog_image = np.abs(blurred_image1 - blurred_image2)

    # Преобразуем значения в диапазон [0, 255] и приводим к типу uint8
    dog_image = np.clip(dog_image, 0, 255).astype(np.uint8)
    return dog_image


def apply_custom_laplacian_of_gaussian_filter(image, sigma):
    # Применяем фильтр Гаусса
    blurred_image = apply_custom_gaussian_filter(image, sigma)

    # Фильтр Лапласа
    laplacian_filter = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    filtered_image = convolve2d(blurred_image, laplacian_filter, mode='same', boundary='symm')

    return filtered_image.astype(np.uint8)


def process_video(video_path, filter_size, frame_step=10):
    # Чтение видео
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print("Total frames:", frame_count)

    # Чтение каждого frame_step-го кадра из видео и обработка
    for i in range(0, frame_count, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Применение функции foo к кадру
            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_grayscale = apply_custom_sobel_filter(frame_grayscale, filter_size)
            # Отображение кадра
            cv2.imshow("Processed Video", frame_grayscale)
            cv2.waitKey(1)  # Необходимо для корректного отображения кадра
        else:
            print("Failed to read frame")

    # Закрытие видео-файла
    cap.release()


def compare_result(image, methods, filter_sizes):
    fig, axes = plt.subplots(len(methods), len(filter_sizes), figsize=(10, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Добавляем отступы между изображениями

    for method_idx, method in enumerate(methods):
        for row_idx in range(len(filter_sizes[method])):
            filter_size = filter_sizes[method][row_idx]
            start_time = time.time()
            output_image = method(image, filter_size)
            end_time = time.time()
            execution_time = end_time - start_time

            ax = axes[method_idx][row_idx]
            ax.imshow(output_image, cmap='gray')
            ax.set_title(f"{method.__name__}\nFilter Size: {filter_size}\nExecution Time: {execution_time:.4f} sec",
                         fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # input_video_path = 'Пол кило.mp4'
    input_video_path = 'Кот кушает2.mp4'
    frame_step = 1  # каждый N-й кадр будет обработан
    filter_size = 5
    process_video(input_video_path, filter_size, frame_step)


if __name__ == "__main__":
    input_image_path = "images/flower.jpg"
    # Чтение изображения с помощью cv2
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    filter_sizes_dict = {apply_custom_sobel_filter: [3, 5, 7], apply_custom_laplacian_of_gaussian_filter: [0.5, 2, 5],
                         apply_difference_of_gaussian_filter: [0.5, 1, 2]}
    methods_filter = [apply_custom_sobel_filter, apply_custom_laplacian_of_gaussian_filter,
                      apply_difference_of_gaussian_filter]
    compare_result(input_image, methods_filter, filter_sizes_dict)

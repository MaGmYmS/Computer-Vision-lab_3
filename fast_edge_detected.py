import cv2
import numpy as np
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


if __name__ == "__main__":
    input_image_path = "images/flower.jpg"
    # Чтение изображения с помощью cv2
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    out_image = apply_difference_of_gaussian_filter(input_image, 1)

    # Отображение изображения
    cv2.imshow("Filtered image", out_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Сохранение изображения
    output_image_path = "sobel_result.jpg"
    cv2.imwrite(output_image_path, out_image)

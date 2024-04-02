
'''
    Main

    File: main.py
    Author: Aidin Lehrman
    Version: 03-28-2024

    Refernces:
        Training Data: https://github.com/zhaodelong/machine-learning-classify-handwritten-digit/tree/master/randomForest
'''

# import matplotlib.pyplot as plt
import numpy as np
from math import e, sqrt
import random
import csv

shading = '.░▒▓█'

def main():
    '''
    with open('./data/digits/train.csv', 'rt', encoding='utf-8') as file:
        file.readline()
        for _ in range(100):
            image: list[int] = map(int, file.readline().split(','))
            print_image(image)
    '''

    # generate_training_data(u'./data/shapes', 10, 10, 10)

    '''
    for i in range(10):
        with open(f'./data/shapes/shape_{i}.csv', 'r', encoding='utf-8') as file:
            image = []
            for line in file:
                image.append(map(lambda value: float(value) / 255, line.split(',')))
            print_image(image)
    '''

    image = generate_circle(50, 50)

    # [print(line) for line in image]

    print_image(image)

def generate_training_data(folder: str, n: int, image_width: int, image_height: int) -> None:
    for i in range(n):
        image: list[list[float]] = []
        if (random.randint(0, 1) == 0):
            image = generate_circle(image_width, image_height)
        else:
            image = generate_rectangle(image_width, image_height)
        
        image = [[int(value) for value in line] for line in image]

        np.savetxt(f'{folder}/shape_{i}.csv', image, delimiter=',')

def generate_rectangle(image_width: int, image_height: int) -> list[list[float]]:
    min_height: int = 2
    min_width: int = 3

    point_top_left: tuple[int, int] = (
        np.random.randint(0, image_width - min_width), 
        np.random.randint(0, image_height - min_height)
    )

    point_bottom_right: tuple[int, int] = (
        np.random.randint(point_top_left[0] + min_width, image_width + 1),
        np.random.randint(point_top_left[1] + min_height, image_height + 1)
    )

    image = np.zeros((image_height, image_width))

    for y in range(point_top_left[1], point_bottom_right[1]):
        for x in range(point_top_left[0], point_bottom_right[0]):
            image[y][x] = 1.0

    return image

def generate_circle(image_width: int, image_height: int) -> list[list[float]]:
    min_radius: int = 3

    # Randomly generate the center point of the circle
    center_point: tuple[int, int] = (
        np.random.randint(min_radius, image_width - min_radius),
        np.random.randint(min_radius, image_height - min_radius)
    )

    # Randomly generate the radius within the allowable range
    radius = np.random.randint(min_radius, 10)

    # Create an empty image
    image = np.zeros((image_height, image_width))

    # Fill the circle in the image
    for y in range(image_height):
        for x in range(image_width):
            # Check if the point (x, y) is inside the circle
            distance_from_center: float = sqrt((center_point[0] - x) ** 2 + (center_point[1] - y) ** 2)

            if (distance_from_center < radius):
                image[y][x] = 1.0
                continue

            scaled_value: float = min(1.0 / (distance_from_center - radius), 1.0)

            image[y][x] = scaled_value

    return image

def print_image(image: list[list[float]]) -> None:
    for line in image:
        for value in line:
            scaled_value: int = int(value * (len(shading) * (1 - 1e-10)))
            print(f'{shading[scaled_value] * 2}', end='')

        print('\n', end='')
    print('\n', end='')

def sigmoid(value: float) -> float:
    ''' Constrains all values between [0.0, 1.0] using Sigmoid function
    '''
    value = 1 / (1 + (e ** -value))
    return value

if __name__ == '__main__':
    main()
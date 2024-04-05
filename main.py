
'''
    Main

    File: main.py
    Author: Aidin Lehrman
    Version: 03-28-2024

    Refernces:
        Training Data: https://github.com/zhaodelong/machine-learning-classify-handwritten-digit/tree/master/randomForest
'''

# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import random

def main():
    n: int = 100
    generate_training_data(u'./training_data/shapes/', n = n, image_width = 50, image_height = 50)


    print(f'Saving Training Data Images to \'./training_data/shapes\' ...')
    for i in range(n):
        with open(f'./training_data/shapes/shape_{i}.csv', 'r', encoding='utf-8') as file:
            image_data: list[list[float]] = read_image_file(file)

            # Convert the list elements to integers and map values to [0, 255] range
            image_data = list(
                map(
                    lambda line: list(
                        map(
                            lambda value: int(math.floor(value * 255)), line
                        )
                    ),
                    image_data
                )
            )

            # Convert the NumPy array to PIL Image and save as PNG
            Image.fromarray(np.array(image_data).astype('uint8')).save(f'./training_data/shapes/images/image_{i}.png')

            # print_image(image)

            print(f'\t{math.floor((i + 1) / n * 100): .0f}%\t[{"■" * int((i / n) * 50)}{" " * (50 - int((i / n) * 50) - 1)}]\t({i + 1}/{n})\t\t', end='\r')
    
    print('\n\n', end='')
        

def generate_training_data(folder: str, *, n: int, image_width: int, image_height: int) -> None:
    print(f'Generating Training Data {{n = {n}, image_width = {image_width}, image_height = {image_height}}} ...')

    with open(u'./training_data/shapes/answers.csv', 'w', encoding='utf-8') as answers:
        answers.write('File_Name, Shape_Name\n')
    
    for i in range(n):
        image: list[list[float]] = []
        shape = ''
        if (random.randint(0, 1) == 0):
            image = generate_circle(image_width, image_height)
            shape = 'circle'
        else:
            image = generate_rectangle(image_width, image_height)
            shape = 'rectangle'
        
        image = [[int(value) for value in line] for line in image]

        np.savetxt(f'{folder}/shape_{i}.csv', image, delimiter=',')

        with open(u'./training_data/shapes/answers.csv', 'a', encoding='utf-8') as answers:
            answers.write(f'shape_{i}.csv, {shape}\n')

        print(f'\t{math.floor((i + 1) / n * 100): .0f}%\t[{"■" * int((i / n) * 50)}{" " * (50 - int((i / n) * 50) - 1)}]\t({i + 1}/{n})\t\t', end='\r')
    
    print('\n\n', end='')

def generate_rectangle(image_width: int, image_height: int) -> list[list[float]]:
    min_height: int = 3
    min_width: int = 10

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
    max_radius: int = 20

    # Randomly generate the radius within the allowable range
    radius = np.random.randint(min_radius, max_radius)

    # Randomly generate the center point of the circle
    center_point: tuple[int, int] = (
        np.random.randint(radius, image_width - radius),
        np.random.randint(radius, image_height - radius)
    )

    # Create an empty image
    image = np.zeros((image_height, image_width))

    # Fill the circle in the image
    for y in range(image_height):
        for x in range(image_width):
            # Check if the point (x, y) is inside the circle
            distance_from_center: float = math.sqrt((center_point[0] - x) ** 2 + (center_point[1] - y) ** 2)

            if (distance_from_center <= radius):
                image[y][x] = 1.0
                continue
            
            '''
            scaled_value: float = min(1.0 / (distance_from_center - radius), 1.0)
            image[y][x] = scaled_value
            '''

    return image

def read_image_file(file) -> list[list[float]]:
    image = []
    for line in file:
        image.append(map(lambda value: float(value), line.split(',')))
    return image

def print_image(image: list[list[float]], shading: str = '.░▒▓█') -> None:
    for line in image:
        for value in line:
            scaled_value: int = int(value * (len(shading) * (1 - 1e-10)))
            print(f'{shading[scaled_value] * 2}', end='')

        print('\n', end='')
    print('\n', end='')

def sigmoid(value: float) -> float:
    ''' Constrains all values between [0.0, 1.0] using Sigmoid function
    '''
    value = 1 / (1 + (math.e ** -value))
    return value

if __name__ == '__main__':
    main()
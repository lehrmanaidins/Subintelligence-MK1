
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
from math import e
import random
import csv

shading = ' ░▒▓█'

def main():
    '''
    with open('./data/digits/train.csv', 'rt', encoding='utf-8') as file:
        file.readline()
        for _ in range(100):
            image: list[int] = map(int, file.readline().split(','))
            print_image(image)
    '''

    # generate_training_data(u'./data/shapes', 1, 28, 28)

    with open('./data/digits/digit_0.csv', 'r', encoding='utf-8') as file:
        image = []
        for line in file:
            image.append(map(lambda value: float(value) / 255, line.split(',')))
        print_image(image)

    # weights: list[list[float]] = np.zeros((28, 28))

def generate_training_data(folder: str, n: int, image_width: int, image_height: int) -> None:
    for i in range(n):
        image: list[list[float]] = [[]]
        if (random.randint(0, 1) == 0):
            image = generate_circle(image_width, image_height)
        else:
            image = generate_rectangle(image_width, image_height)
        
        np.savetxt(f'{folder}/shape_{i}.csv', image, delimiter=',')

def generate_rectangle(image_width: int, image_height: int) -> list[list[float]]:
    image = np.random.rand(image_width, image_height)
    return image

def generate_circle(image_width: int, image_height: int) -> list[list[float]]:
    image = np.random.rand(image_width, image_height)
    return image

def print_image(image: list[list[float]]) -> None:
    for line in image:
        for value in line:
            scaled_value: int = int(value * (len(shading) * (1 - 1e-10)))
            print(f'{shading[scaled_value] * 2}', end='')

        print('\n', end='')

def sigmoid(value: float) -> float:
    ''' Constrains all values between [0.0, 1.0] using Sigmoid function
    '''
    value = 1 / (1 + (e ** -value))
    return value

if __name__ == '__main__':
    main()

'''
    Main

    File: main.py
    Author: Aidin Lehrman
    Version: 03-28-2024

    Refernces:
        Training Data: https://github.com/zhaodelong/machine-learning-classify-handwritten-digit/tree/master/randomForest
'''

import matplotlib.pyplot as plt
import numpy as np
from math import e

def main():
    with open('./data/train.csv', 'rt', encoding='utf-8') as file:
        file.readline()
        for _ in range(100):
            image: list[int] = map(int, file.readline().split(','))
            print_image(image)
    
    # weights: list[list[float]] = np.zeros((28, 28))

def print_image(image: list[int]) -> None:
    for i, value in enumerate(image):
        if (i % 28 == 0):
            print('\n', end='')
        if (value >= 200):
            print('X', end='')
        else:
            print(' ', end='')

def sigmoid(value: float) -> float:
    ''' Constrains all values between [0.0, 1.0] using Sigmoid function
    '''
    value = 1 / (1 + (e ** -value))
    return value

if __name__ == '__main__':
    main()
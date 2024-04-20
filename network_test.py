
'''
    Neural Network Test 

    File network_test.py
    Author: Aidin Lehrman
    Version: 04-20-2024 08:24
'''


import numpy as np
from data_generator import generate_training_data


def test_network(test_data: list[list[float]], answers_csv_file: str, weight_csv_file: str) -> float:

    weights: list[float] = []

    with open(weight_csv_file, 'r', encoding='utf-8') as weights_file:
        for line in weight_csv_file:
            row = [float(data.strip()) for data in line.split(',')]
            weights.append(row)

    for image in test_data:
        pass


def main():
    image_width: int = 50
    image_height: int = 50
    num_samples: int = 100

    test_data: list[list[list[float]]] = generate_training_data(u'./test_answers.csv', n = num_samples, image_width = image_width, image_height = image_height)
    flattened_test_data: list[list[float]] = [np.reshape(image, image_width * image_height) for image in test_data]

    test_network(flattened_test_data, u'./test_answers.csv', u'./weights.csv')


if __name__ == '__main__':
    main()

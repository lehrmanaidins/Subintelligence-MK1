
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
        weights.append([float(data.strip()) for data in weights_file.readline().split(', ')])

    with open(answers_csv_file, 'r', encoding='utf-8') as answers_file:
        for image in test_data:
            answer = answers_file.readline().strip()

            raw_neural_network_output = np.dot(image, weights[0])
            cleaned_neural_network_output = min(1.0, max(0.0, raw_neural_network_output))

            desired_output: int = 1.0 if (answer == 'circle') else 0.0
            
            print(f'Output: {cleaned_neural_network_output:.1f}, Desired: {desired_output:.1f}, Passed: {cleaned_neural_network_output == desired_output}')

def main():
    image_width: int = 20
    image_height: int = 20
    num_samples: int = 10

    test_data: list[list[list[float]]] = generate_training_data(u'./test_answers.csv', n = num_samples, image_width = image_width, image_height = image_height)
    flattened_test_data: list[list[float]] = [np.reshape(image, image_width * image_height) for image in test_data]

    test_network(flattened_test_data, u'./test_answers.csv', u'./weights.csv')


if __name__ == '__main__':
    main()

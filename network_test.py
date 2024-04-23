
'''
    Neural Network Test 

    File network_test.py
    Author: Aidin Lehrman
    Version: 04-20-2024 08:24
'''


import numpy as np
from data_generator import generate_training_data


def test_network(test_data, answers_csv_file: str, weight_csv_file: str) -> float:

    weights: list[float] = []

    with open(weight_csv_file, 'r', encoding='utf-8') as weights_file:
        weights = [float(data.strip()) for data in weights_file.readline().split(', ')]

    guessed_correctly: int = 0

    with open(answers_csv_file, 'r', encoding='utf-8') as answers_file:
        for image in test_data:
            answer = answers_file.readline().strip()

            raw_neural_network_output = np.dot(image, weights)
            
            cleaned_neural_network_output = min(1.0, max(0.0, raw_neural_network_output))

            desired_output: float = 1.0 if (answer == 'circle') else 0.0
            
            if (desired_output == cleaned_neural_network_output):
                guessed_correctly += 1
            
    return guessed_correctly / len(test_data)


def main() -> None:
    image_width: int = 20
    image_height: int = 20
    num_samples: int = 100

    test_data = np.array(generate_training_data(u'./test_answers.csv', n = num_samples, image_width = image_width, image_height = image_height))
    flattened_test_data = [list(np.reshape(image, image_width * image_height)) for image in test_data]

    percent_guessed_correctly: float = test_network(flattened_test_data, u'./test_answers.csv', u'./weights.csv')

    print(f'Percent Guessed Correctly: {percent_guessed_correctly * 100:.2f}%')

if __name__ == '__main__':
    main()

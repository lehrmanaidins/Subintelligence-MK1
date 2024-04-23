
'''
    Circle and Rectangle Identifying Neural Network Training

    File: network_training.py
    Author: Aidin Lehrman
    Version: 04-19-2024 10:25

    References:
        https://docs.google.com/document/d/1SnwIpVScZHWpri4EMlm59nHwJLI09HBUE5iGSabQjfw/edit
'''


import numpy as np
import os
from data_generator import generate_training_data
import threading


image_width = 20
image_height = 20

weights = np.zeros(image_width * image_height)

def train(answers_file_path: str, training_data):

    global weights

    with open(answers_file_path, 'r', encoding='utf-8') as answers_file:

        guessed_correctly: int = 0

        for image, desired_output_str in zip(training_data, answers_file):

            raw_neural_network_output = np.dot(image, weights)
            cleaned_neural_network_output = min(1.0, max(0.0, raw_neural_network_output))

            desired_output = 1.0 if desired_output_str.strip() == 'circle' else 0.0

            error = desired_output - cleaned_neural_network_output 

            if (desired_output == cleaned_neural_network_output):
                guessed_correctly += 1

            weights += np.multiply(image, error)

    with open('./weights.csv', 'w', encoding='utf-8') as weight_file:
        weight_file.write(str(list(weights))[1: -1])

    return guessed_correctly / len(training_data)


def train_cycles(answers_file_path, images, max_cycles: int) -> None:
    
    # It is necessary to cycle through and train with the same samples a couple of times.
    for i in range(max_cycles):
        percent_guess_correctly: float = train(answers_file_path, images)

        print(f'\r\tTraining ... (#{i + 1}), Max Cycles: {max_cycles}. Percent Accuracy: {percent_guess_correctly * 100:.2f}%', end='')

        # Passed most tests
        if (percent_guess_correctly > 0.9999):
            break
        
    print('\nDone Training.')
    

def main() -> None:
    num_samples = 10_000

    answers_file_path = os.path.join('.', 'training_answers.csv')
    
    training_data = generate_training_data(answers_file_path, n=num_samples, image_width=image_width, image_height=image_height)
    flattened_training_data = [list(np.reshape(image, image_width * image_height)) for image in training_data]

    threads = []

    for _ in range(10):
        thread = threading.Thread(target=train_cycles, args=(answers_file_path, flattened_training_data, 50))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
 
    print("Done!")


if __name__ == "__main__":
    main()

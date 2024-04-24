
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

weights: list[float] = list(np.zeros(image_width * image_height))

average_accuracy: float = 0.0

def train(answers_list: list[str], training_data):
    
    guessed_correctly: int = 0

    for image, desired_output_str in zip(training_data, answers_list):
        
        global weights # Gets updated weight values

        raw_neural_network_output = np.dot(image, weights)
        cleaned_neural_network_output = min(1.0, max(0.0, raw_neural_network_output))

        desired_output = 1.0 if desired_output_str.strip() == 'circle' else 0.0

        error = desired_output - cleaned_neural_network_output 

        if (desired_output == cleaned_neural_network_output):
            guessed_correctly += 1

        weights += np.multiply(image, error)

    return guessed_correctly / len(training_data)


def train_cycles(answers_file_path, images, max_cycles: int) -> None:
    
    # It is necessary to cycle through and train with the same samples a couple of times.
    for i in range(max_cycles):
        percent_guess_correctly: float = train(answers_file_path, images)
        
        global average_accuracy
        average_accuracy = (percent_guess_correctly + average_accuracy) / 2

        if (i % 50 == 0):
            print(f'\r\tTraining ... (#{i + 1:0>4}), Max Cycles: {max_cycles}. Percent Accuracy: {average_accuracy * 99.99:>6.2f}%', end='')
    

def main() -> None:
    
    global weights
    
    # Get weight values stored in 'weights.csv' file
    with open('./weights.csv', 'r', encoding='utf-8') as weight_file:
        weights = [float(value.strip()) for value in weight_file.readline().strip().split(', ')]
    
    num_samples = 10_000

    answers_file_path = os.path.join('.', 'training_answers.csv')
    
    training_data = generate_training_data(answers_file_path, n=num_samples, image_width=image_width, image_height=image_height)
    flattened_training_data = [list(np.reshape(image, image_width * image_height)) for image in training_data]
    
    answers_list: list[str] = []
    with open(answers_file_path, 'r', encoding='utf-8') as answers_file:
        for line in answers_file:
            answers_list.append(line)

    threads = []

    num_threads: int = 100
    max_num_cycles: int = 1_000
    
    for i in range(num_threads):
        thread = threading.Thread(
            target=train_cycles,
            args=(
                answers_list[i * int(num_samples / num_threads): (i + 1) * int(num_samples / num_threads)],
                flattened_training_data[i * int(num_samples / num_threads): (i + 1) * int(num_samples / num_threads)],
                max_num_cycles
            )
        )
        threads.append(thread)
        
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
        
    with open('./weights.csv', 'w', encoding='utf-8') as weight_file:
        weight_file.write(str(list(weights))[1:-1])
 
    print("\nDone Training!")


if __name__ == "__main__":
    try:
        main()
        
    except (InterruptedError): # Saves weight values if InterruptedError exception occurs
        pass
    finally:
        with open('./weights.csv', 'w', encoding='utf-8') as weight_file:
            weight_file.write(str(list(weights))[1:-1])


'''
    Circle and Rectangle Identifying Neural Network Training

    File: network_training.py
    Author: Aidin Lehrman
    Version: 04-19-2024 10:25

    References:
        https://docs.google.com/document/d/1SnwIpVScZHWpri4EMlm59nHwJLI09HBUE5iGSabQjfw/edit
'''

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
import os
from data_generator import generate_training_data

image_width = 50
image_height = 50
output_neurons = 1

weights = np.zeros((output_neurons, image_width * image_height))


def train(training_data):

    answers_file_path = os.path.join('.', 'training_data', 'shapes', 'answers.csv')

    with open(answers_file_path, 'r', encoding='utf-8') as answers_file:

        header = answers_file.readline()

        for image, desired_output_str in zip(training_data, answers_file):

            raw_neural_network_output = np.dot(image, weights[0])
            cleaned_neural_network_output = round(sigmoid(raw_neural_network_output))

            desired_output = 1.0 if desired_output_str.split(',')[1].strip() == 'circle' else 0.0

            error = desired_output - cleaned_neural_network_output

            weights[0] += error * image

    with open('./weights.csv', 'w', encoding='utf-8') as weight_file:
        weight_file.write(str(list(weights[0]))[1: -1])

    fig, ax = plt.subplots()
    ax.pcolormesh(np.arange(image_width), np.arange(image_height), np.reshape(weights[0], (image_width, image_height)))
    plt.show()


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def main():
    num_samples = 1_000

    answers_file_path = os.path.join('.', 'training_data', 'shapes', 'answers.csv')
    training_data = generate_training_data(answers_file_path, n=num_samples, image_width=image_width, image_height=image_height)

    training_data = [np.reshape(image, image_width * image_height) for image in training_data]

    train(training_data)


if __name__ == "__main__":
    main()

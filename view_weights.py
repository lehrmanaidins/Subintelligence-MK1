
'''
    View Weights

    File: view_weights.py
    Author: Aidin Lehrman
    Version: 04-23-2024 09:35
'''


import matplotlib.pyplot as plt
import numpy as np


def view_weights(weights: list[float], image_width: int = 20, image_height: int = 20) -> None:
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(np.arange(image_width), np.arange(image_height), np.reshape(weights, (image_width, image_height)))
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Pixel Values')
    plt.show()


def main() -> None: 
    image_width, image_height = (20, 20)

    weights: list[float] = []

    with open(u'weights.csv', 'r', encoding='utf-8') as weights_file:
        weights = [float(data.strip()) for data in weights_file.readline().split(', ')]
                   
    view_weights(weights, image_width, image_height)


if __name__ == '__main__':
    main()

# Sub-Intelligence Mark-I
This neural network is designed to be able to take an 20x20 image and detect whether the inputed image contains either a rectangle or a circle.

## Neural Network Structure
### General Network Layer Value Calculator
$y = W^Tx$

 - $y$ is new neuron layer values vector.
$$y =
\left[ {\begin{array}{c}
y_{1} \\
y_{2} \\
\vdots \\
y_{n} \\
\end{array} } \right]
$$

- $W$ is the weight values matrix.
$$
W_{n\times n} =
\left[ {\begin{array}{cccc}
w_{1,1} & w_{1,2} & \cdots & w_{1,n}\\
w_{2,1} & w_{2,2} & \cdots & w_{2,n}\\
\vdots & \vdots & \ddots & \vdots\\
w_{n,1} & w_{n,2} & \cdots & w_{n,n}\\
\end{array} } \right]
$$

- and $x$ is the previous/input neuron values vector.
$$
x =
\left[ {\begin{array}{c}
x_{1} \\
x_{2} \\
\vdots \\
x_{n} \\
\end{array} } \right]
$$


## "The Great 70% Challenge"
The neural network seems to struggle to get an accuracy of guessing the shapes over 70% no matter how many images 
are used to train. I have tried to train it with both 10,000 and 100,000 images, and, in both scenarios, the network 
is correctly guessing the test shapes about 70% of the time. 

I have two theories as to why this might be the case. Either:
- [ ] Only 70% of the test images generated are close to/the same as the training images.
- [ ] When there are a large number of images used to train the network weights, the weights seem to "blend"
    together and the network can only correctly guess about 70% of the time.

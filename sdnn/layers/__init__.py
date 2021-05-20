import numpy as np


class Layer:
    """Layer abstract class"""

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass

    def forward(self, input_feature):
        pass

    def backward(self, d_a):
        pass

    def optimize(self):
        pass


class Linear(Layer):
    """Linear.

    A linear layer. Equivalent to Dense in Keras and to torch.nn.Linear
    in torch.

    Parameters
    ----------
    input_dim : int
        Number of input features of this layer.
    output_dim : int
        Number of output features of this layer.

    """

    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(output_dim, input_dim)
        self.biases = np.random.rand(1, output_dim)
        self.units = output_dim
        self.type = 'Linear'
        self._prev_act = None
        self.grad = None

    def _len_(self):
        return self.units

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        """Forward.

        Performs forward propagation of this layer.

        Parameters
        ----------
        input_val : numpy.Array
            Forward propagation of the previous layer.

        Returns
        -------
        activation : numpy.Array
            Forward propagation operation of the linear layer.

        """
        self._prev_act = input_val
        # return np.matmul(input_val, np.transpose(self.weights)) + self.biases
        return np.matmul(input_val, self.weights.T)

    def backward(self, dA):
        """Backward.

        Performs backward propagation of this layer.

        Parameters
        ----------
        dA : numpy.Array
            Gradient of the next layer.

        Returns
        -------
        delta : numpy.Array
            Upcoming gradient, usually from an activation function.
        dW : numpy.Array
            Weights gradient of this layer.
        dB : numpy.Array
            Biases gradient of this layer.
        References
        ----------
        [1] Justin Johnson - Backpropagation for a Linear Layer:
        http://cs231n.stanford.edu/handouts/linear-backprop.pdf
        [2] Pedro Almagro Blanco - Algoritmo de Retropropagación:
        http://www.cs.us.es/~fsancho/ficheros/IAML/2016/Sesion04/
        capitulo_BP.pdf
        [3] Raúl Rojas - Neural Networks: A Systematic Introduction:
        https://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf

        """
        # dA: 1 * 2  input: 1 * 3
        print(dA.T.shape)
        print(self._prev_act.shape)
        dW = np.matmul(dA.T, self._prev_act)
        dB = None
        self.grad = dW

        # dA: 1 * 2  weight: 2 * 3
        delta = np.matmul(dA, self.weights)

        return delta, dW, dB

    def optimize(self, dW, dB, rate):
        """Optimizes.

        Performs the optimization of the parameters. For now,
        optimization can only be performed using gradient descent.

        Parameters
        ----------
        dW : numpy.Array
            Weights gradient.
        dB : numpy.Array
            Biases gradient.
        rate: float
            Learning rate of the gradient descent.

        """
        self.weights = self.weights - rate * dW
        self.biases = self.biases - rate * dB

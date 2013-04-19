import numpy as np


def sigmoid(vector):
    return 1. / (1 + np.exp(-vector))


class MPL():
    def __init__(self, input_layer_size=1, output_layer_size=1, hidden_layers_sizes=None, random_init_epsilon=0.1):
        if hidden_layers_sizes is None:
            hidden_layers_sizes = []

        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.layers_sizes = [input_layer_size] + hidden_layers_sizes + [output_layer_size]

        #se initial random weights from interval [-epsilon, epsilon]
        self.weights = []
        for i in range(len(self.layers_sizes) - 1):
            weights = (np.random.rand(self.layers_sizes[i] + 1, self.layers_sizes[i + 1]) - .5) * 2 * random_init_epsilon
            self.weights.append(weights)

    def compute(self, input_vector, debug=False):
        """ Forward propagation"""

        if type(input_vector) is not np.ndarray:
            raise ValueError("Input vector must be Numpy Array")

        if input_vector.size != self.input_layer_size:
            raise ValueError("Input vector must have size "
                            + str(self.input_layer_size)
                            + " (not "
                            + str(input_vector.size)
                            + ")")

        vector = input_vector
        self.neurons_potential = [None]
        self.neurons_value = [vector]
        for w in self.weights:
            vector = np.dot(w.T, np.insert(vector, 0, 1))
            self.neurons_potential.append(vector)
            vector = self.activation_function(vector)
            self.neurons_value.append(vector)
            if debug:
                print vector

        return vector

    def backpropagate(self, data, version=0, epsilon=1, reg_lambda=0.5):
        """Update weight based on data

        :param data: list of tuples (input vector, expected result)
        :param version: 1 - for network from coursea.com course by Ng, 0 - for network from slides
        :param epsilon: for version 0 - learning speed
        :param reg_lambda: for regularization
        """

        #init weight difference
        deltas = []
        for w in self.weights:
            deltas.append(np.zeros(w.shape))

        for input_vector, expected_vector in data:

            # check input values
            if type(expected_vector) is not np.ndarray:
                raise ValueError("Output vector must be Numpy Array")

            if expected_vector.size != self.output_layer_size:
                raise ValueError("Output vector must have size "
                                + str(self.output_layer_size)
                                + " (not "
                                + str(expected_vector.size)
                                + ")")

            # forward propagation
            self.compute(input_vector)
            # compute errors on neurons
            self.compute_errors(expected_vector)

            # update delta
            for i, delta in enumerate(deltas):
                if version == 1:
                    delta += np.outer(np.insert(self.neurons_value[i], 0, 1), self.errors[i + 1])
                else:
                    delta += np.outer(np.insert(self.neurons_value[i], 0, 1),
                                  self.errors[i + 1] * self.diff_of_activation_function(self.neurons_potential[i + 1]))

        # change weights
        for i, delta in enumerate(deltas):
            d = (1. / len(data)) * epsilon * delta
            if reg_lambda:
                d[1:] += (reg_lambda / len(data)) * self.weights[i][1:]

            self.weights[i] -= d

    def compute_errors(self, expected_vector):
        """ backpropagate errors"""

        self.errors = [None] * len(self.layers_sizes)

        error = self.neurons_value[-1] - expected_vector
        self.errors[-1] = error

        for i in range(len(self.weights) - 1, 0, -1):
            error = np.dot(self.weights[i], error)[1:] * self.diff_of_activation_function(self.neurons_potential[i])
            self.errors[i] = error

    def activation_function(self, vector):
        return sigmoid(vector)

    def diff_of_activation_function(self, vector):
        """derivation of activation function"""

        s = sigmoid(vector)
        return s * (1 - s)

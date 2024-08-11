"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np

# my libraries
import NetworkTools

#### Define the quadratic and cross-entropy cost functions


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        # return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
        return np.sum(np.nan_to_num(-y * np.log(a + 0.000001) - (1 - y) * np.log(1 - a * 0.999999)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return a - y


#### Main Network class
class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def feedforwardPrint(self, a):
        """Return the output of the network if ``a`` is input."""
        n = len(a)
        for b, w in zip(self.biases, self.weights):
            aux = np.dot(w, a).reshape((80, n))
            a = sigmoid(aux + b)
        # no need to NORMALIZE
        # PRINT
        for i in range(a.shape[-1]):
            print(
                np.argmax(a[0:10, i]),
                np.argmax(a[10:20, i]),
                np.argmax(a[20:30, i]),
                np.argmax(a[30:40, i]),
                "   ",
                np.argmax(a[40:50, i]),
                np.argmax(a[50:60, i]),
                np.argmax(a[60:70, i]),
                np.argmax(a[70:80, i]),
            )
            print()

    def feedforwardPrintExtended(self, a):
        """Return the output of the network if ``a`` is input."""
        n = len(a)
        for b, w in zip(self.biases, self.weights):
            aux = np.dot(w, a).reshape((80, n))
            a = sigmoid(aux + b)
        # NORMALIZE
        for i in range(a.shape[-1]):
            for j in range(8):
                sum = np.sum(a[j * 10 : (j + 1) * 10, i])
                a[j * 10 : (j + 1) * 10, i] = a[j * 10 : (j + 1) * 10, i] / sum
        # PRINT
        np.set_printoptions(formatter={"float": "{: 0.2f}".format})
        aux = np.array([float(x) for x in range(10)])
        for i in range(a.shape[-1]):
            print(aux, aux)
            print(a[0:10, i], a[40:50, i])
            print(a[10:20, i], a[50:60, i])
            print(a[20:30, i], a[60:70, i])
            print(a[30:40, i], a[70:80, i])
            print()

    def SGD(
        self,
        training_data,
        max_epochs,
        mini_batch_size,
        eta,
        lmbda = 0.0,
        evaluation_data = None,
        eta_change_interval = 10,
        eta_change_factor = 0.5
    ):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy = 0
        prev_eta_best_accuracy = 0
        best_accuracy_nr_epochs = 0
        current_epoch = 0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        while current_epoch <= max_epochs and best_accuracy_nr_epochs < eta_change_interval + 1:
            current_epoch += 1


            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            # print("Epoch %s training complete" % j)

            cost = self.total_cost(training_data, lmbda)
            training_cost.append(cost)
            # print("Cost on training data: {}".format(cost))

            accuracy = self.accuracy(training_data)
            training_accuracy.append(accuracy)
            # print("Accuracy on training data: {} / {}".format(accuracy, 1))  # n instead of 1
            
            cost = self.total_cost(evaluation_data, lmbda)
            evaluation_cost.append(cost)
            # print("Cost on evaluation data: {}".format(cost))
            
            accuracy = self.accuracy(evaluation_data)
            evaluation_accuracy.append(accuracy)
            # print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), 1))  # n_data instead of 1

            # Early stopping:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_nr_epochs = 0
                # print("Early-stopping: Best so far {}".format(best_accuracy))
            else:
                best_accuracy_nr_epochs += 1

            if best_accuracy_nr_epochs == eta_change_interval + 1:
                if prev_eta_best_accuracy < best_accuracy:
                    prev_eta_best_accuracy = best_accuracy
                    best_accuracy_nr_epochs = 0
                    eta = eta * eta_change_factor
                else:
                    # print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return (
                        evaluation_cost,
                        evaluation_accuracy,
                        training_cost,
                        training_accuracy,
                    )

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data
            ]
        else:
            results = [(self.feedforward(x), y) for (x, y) in data]

        result_accuracy = 0
        for (x, y) in results:
            result_accuracy = result_accuracy + int(
                np.argmax(x[0:10]) == np.argmax(y[0:10])
            )
            result_accuracy = result_accuracy + int(
                np.argmax(x[10:20]) == np.argmax(y[10:20])
            )
            result_accuracy = result_accuracy + int(
                np.argmax(x[20:30]) == np.argmax(y[20:30])
            )
            result_accuracy = result_accuracy + int(
                np.argmax(x[30:40]) == np.argmax(y[30:40])
            )
            result_accuracy = result_accuracy + int(
                np.argmax(x[40:50]) == np.argmax(y[40:50])
            )
            result_accuracy = result_accuracy + int(
                np.argmax(x[50:60]) == np.argmax(y[50:60])
            )
            result_accuracy = result_accuracy + int(
                np.argmax(x[60:70]) == np.argmax(y[60:70])
            )
            result_accuracy = result_accuracy + int(
                np.argmax(x[70:80]) == np.argmax(y[70:80])
            )
        return result_accuracy / len(data) / 8

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = NetworkTools.vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
            cost += (
                0.5
                * (lmbda / len(data))
                * sum(np.linalg.norm(w) ** 2 for w in self.weights)
            )  # '**' - to the power of.
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__),
        }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

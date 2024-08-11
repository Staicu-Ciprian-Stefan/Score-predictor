# standard libraries


# 3rd party libraries
import math
import matplotlib.pyplot as plt

# my libraries
import GlobalParameters


def vectorized_result(input_value):
    return __vectorized_result__(input_value, GlobalParameters.vector_size, GlobalParameters.scale_factor)


def __vectorized_result__(input_value, currect_vector_size = 10, current_scale_factor = 0):
    input_value = int(input_value)
    result = [0] * currect_vector_size
    sum = 0
    for i in range(currect_vector_size):
        result[i] = math.exp(current_scale_factor * (abs(input_value - i)))
        sum = sum + result[i]

    for i in range(currect_vector_size):
        result[i] = result[i] / sum

    return result


def general_final_check(my_list, expected_size, text):
    for item in my_list:
        assert (item.shape[0] == expected_size
        ), "Invalid shape %d, expected %d for %s " % (
            item.shape[0],
            expected_size,
            text
        )
        assert (item.shape[1] == 1
        ), "Invalid shape %d, expected %d for %s " % (
            item.shape[1],
            1,
            text
        )


def draw_stats(training_cost, training_accuracy, evaluation_cost = None, evaluation_accuracy = None, phase = None):
    fig = plt.figure()
    fig.suptitle(phase + ' training stats', fontsize = 16)
    training_cost_graph = fig.add_subplot(221)
    training_cost_graph.plot(training_cost)
    training_cost_graph.grid(True)
    # tc.set_xlabel("Epoch")
    training_cost_graph.set_ylabel("Cost")
    training_cost_graph.set_title("Training Cost")

    training_accuracy_graph = fig.add_subplot(222)
    training_accuracy_graph.plot(training_accuracy)
    training_accuracy_graph.grid(True)
    # tc.set_xlabel("Epoch")
    training_accuracy_graph.set_ylabel("Accuracy %")
    training_accuracy_graph.set_title("Training Accuracy")

    if evaluation_cost != []:
        evaluation_cost_graph = fig.add_subplot(223)
        evaluation_cost_graph.plot(evaluation_cost)
        evaluation_cost_graph.grid(True)
        evaluation_cost_graph.set_xlabel("Epoch")
        evaluation_cost_graph.set_ylabel("Cost")
        evaluation_cost_graph.set_title("Evaluation Cost")

    if evaluation_accuracy != []:
        evaluation_accuracy_graph = fig.add_subplot(224)
        evaluation_accuracy_graph.plot(evaluation_accuracy)
        evaluation_accuracy_graph.grid(True)
        evaluation_accuracy_graph.set_xlabel("Epoch")
        evaluation_accuracy_graph.set_ylabel("Accuracy %")
        evaluation_accuracy_graph.set_title("Evaluation Accuracy")

    if GlobalParameters.is_debug == False:
        training_cost_graph.set_ylim(bottom = 0)
        training_cost_graph.set_xlim(left = 0)
        training_accuracy_graph.set_ylim(bottom = 0, top = 100)
        training_accuracy_graph.set_xlim(left = 0)
        evaluation_cost_graph.set_ylim(bottom = 0)
        evaluation_cost_graph.set_xlim(left = 0)
        evaluation_accuracy_graph.set_ylim(bottom = 0, top = 100)
        evaluation_accuracy_graph.set_xlim(left = 0)

    plt.show()
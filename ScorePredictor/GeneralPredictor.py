import matplotlib.pyplot as plt
import pandas as pd
import Network2


class GeneralPredictor:
    def __init__(self):
        # can't initialize data here because multiple networks are used for each type of prediction
        self._reset_total_cost_accuracy()
    
    def _reset_total_cost_accuracy(self):
        self.total_training_cost = []
        self.total_training_accuracy = []
        self.total_evaluation_cost = []
        self.total_evaluation_accuracy = []

    def _get_network_input_size(self):
        raise NotImplementedError
    
    def _get_network_output_size(self):
        raise NotImplementedError
   
    def train():
        raise NotImplementedError
    
    def _general_check(self, my_list, expected_size, text):
        assert (my_list.shape[0] == expected_size
        ), "Invalid shape %d, expected %d for %s " % (
            my_list[0].shape[0],
            expected_size,
            text
        )
        assert (my_list.shape[1] == 1
        ), "Invalid shape %d, expected %d for %s " % (
            my_list[0].shape[1],
            1,
            text
        )

    def _general_final_check(self, my_list, expected_size, text):
        assert (my_list[0].shape[0] == expected_size
        ), "Invalid shape %d, expected %d for %s " % (
            my_list[0].shape[0],
            expected_size,
            text
        )
        assert (my_list[0].shape[1] == 1
        ), "Invalid shape %d, expected %d for %s " % (
            my_list[0].shape[1],
            1,
            text
        )

    def get_results(self, teams, games):
        output = []
        for team in teams:
            for game in team.team_games:
                x = 0
                # output.append()
        return output

    def _training_wrapper(self, training_input, training_output, evaluation_input, evaluation_output):
        # make checks
        self._general_final_check(training_input, self._get_network_input_size(), "training input final check")
        self._general_final_check(training_output, self._get_network_output_size(), "training output final check")
        self._general_final_check(evaluation_input, self._get_network_input_size(), "evaluation input final check")
        self._general_final_check(evaluation_output, self._get_network_output_size(), "evaluation output final check")
        
        # define network
        net = Network2.Network([self._get_network_input_size(), self._get_network_output_size()], cost = Network2.CrossEntropyCost)
        (
            evaluation_cost,
            evaluation_accuracy,
            training_cost,
            training_accuracy,
        ) = net.SGD(
            zip(training_input, training_output),
            10,  # epochs
            1,  # batch size
            2.0,  # learning rate
            evaluation_data = zip(evaluation_input, evaluation_output),
            monitor_evaluation_cost = True,
            monitor_evaluation_accuracy = True,
            monitor_training_cost = True,
            monitor_training_accuracy = True,
        )
        if self.total_training_cost == []:
            self.total_training_cost = training_cost
        else:
            self.total_training_cost = [
                sum(x) for x in zip(self.total_training_cost, training_cost)
            ]
        if self.total_training_accuracy == []:
            self.total_training_accuracy = training_accuracy
        else:
            self.total_training_accuracy = [
                sum(x) for x in zip(self.total_training_accuracy, training_accuracy)
            ]
        if self.total_evaluation_cost == []:
            self.total_evaluation_cost = evaluation_cost
        else:
            self.total_evaluation_cost = [
                sum(x) for x in zip(self.total_evaluation_cost, evaluation_cost)
            ]
        if self.total_evaluation_accuracy == []:
            self.total_evaluation_accuracy = evaluation_accuracy
        else:
            self.total_evaluation_accuracy = [
                sum(x) for x in zip(self.total_evaluation_accuracy, evaluation_accuracy)
            ]
        return (evaluation_cost, evaluation_accuracy, training_cost, training_accuracy)
        

    def _draw_stats(self, phase):
        fig = plt.figure()
        fig.suptitle(phase + ' stats', fontsize = 16)
        training_cost_graph = fig.add_subplot(221)
        training_cost_graph.plot(self.total_training_cost)
        training_cost_graph.set_ylim(bottom = 0)
        training_cost_graph.set_xlim(left = 0)
        training_cost_graph.grid(True)
        # tc.set_xlabel("Epoch")
        training_cost_graph.set_ylabel("Cost")
        training_cost_graph.set_title("Training Cost")

        training_accuracy_graph = fig.add_subplot(222)
        training_accuracy_graph.plot(self.total_training_accuracy)
        training_accuracy_graph.set_ylim(bottom = 0, top = 100)
        training_accuracy_graph.set_xlim(left = 0)
        training_accuracy_graph.grid(True)
        # tc.set_xlabel("Epoch")
        training_accuracy_graph.set_ylabel("Accuracy %")
        training_accuracy_graph.set_title("Training Accuracy")

        evaluation_cost_graph = fig.add_subplot(223)
        evaluation_cost_graph.plot(self.total_evaluation_cost)
        evaluation_cost_graph.set_ylim(bottom = 0)
        evaluation_cost_graph.set_xlim(left = 0)
        evaluation_cost_graph.grid(True)
        evaluation_cost_graph.set_xlabel("Epoch")
        evaluation_cost_graph.set_ylabel("Cost")
        evaluation_cost_graph.set_title("Evaluation Cost")

        evaluation_accuracy_graph = fig.add_subplot(224)
        evaluation_accuracy_graph.plot(self.total_evaluation_accuracy)
        evaluation_accuracy_graph.set_ylim(bottom = 0, top = 100)
        evaluation_accuracy_graph.set_xlim(left = 0)
        evaluation_accuracy_graph.grid(True)
        evaluation_accuracy_graph.set_xlabel("Epoch")
        evaluation_accuracy_graph.set_ylabel("Accuracy %")
        evaluation_accuracy_graph.set_title("Evaluation Accuracy")

        plt.show()
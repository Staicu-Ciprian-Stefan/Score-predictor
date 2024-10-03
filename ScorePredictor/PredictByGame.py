# standard libraries

# 3rd party libraries
import numpy
import pandas as pd

# my libraries
import Network2
import NetworkTools
import GlobalParameters
import ScoreClass

class PredictByGame:
    def __init__(self):
        self.net = -1
    
    # cele 2 trebuie unificate in functie de is_debug
    def train(self, phases, teams, games):
        training_input = []
        training_output = []

        # get size
        input_size = len(games[0].team1.get_stats(phases[0])) * 2
        output_size = 122

        # instead of reversing the games we will go over each team and its games
        # this way each game will automatically be considered twice, with teams reversed

        # get training data, no evaluation
        for team in teams:
            for game in team.team_games:
                if game.is_played:
                    input_data = numpy.array(game.get_input()).reshape(input_size, 1)
                    output_data = numpy.array(game.get_output()).reshape(output_size, 1)
                    training_input.append(input_data)
                    training_output.append(output_data)

        # make checks
        NetworkTools.general_final_check(
            training_input, input_size, "training input final check"
        )
        NetworkTools.general_final_check(
            training_output, output_size, "training output final check"
        )

        # define network
        self.net = Network2.Network(
            [input_size, output_size], cost = Network2.CrossEntropyCost
        )
    
        # train
        (
            evaluation_cost,
            evaluation_accuracy,
            training_cost,
            training_accuracy,
        ) = self.net.SGD(
            zip(training_input, training_output),
            20,  # max epochs
            1,  # batch size
            2.0,  # learning rate
            0.1, # lambda
            eta_change_interval = 5, # learning rate change interval
            eta_change_factor = 0.5 # learning rate change factor
        )

        # Postprocessing
        NetworkTools.draw_stats(
            training_cost,
            training_accuracy,
            evaluation_cost,
            evaluation_accuracy,
        )

    # cele 2 trebuie unificate in functie de is_debug
    def debug_train(self, phases, teams, games):
        for predicted_phase in phases[1:2]:
            print("Phase " + predicted_phase + " training started.")
            training_nr_games = 0
            evaluation_nr_games = 0
            training_input = []
            training_output = []
            evaluation_input = []
            evaluation_output = []

            # get size
            input_size = len(games[0].team1.get_stats(predicted_phase)) * 2
            output_size = 122

            # instead of reversing the games we will go over each team and its games
            # this way each game will automatically be considered twice, with teams reversed

            # get data
            for team in teams:
                for game in team.team_games:
                    if game.is_played:
                        input_data = numpy.array(game.get_input(predicted_phase)).reshape(input_size, 1)
                        output_data = numpy.array(game.get_output()).reshape(output_size, 1)
                        if game.phase < predicted_phase:
                            training_nr_games += 1
                            training_input.append(input_data)
                            training_output.append(output_data)
                        if game.phase == predicted_phase:
                            evaluation_nr_games += 1
                            evaluation_input.append(input_data)
                            evaluation_output.append(output_data)

            # make checks
            NetworkTools.general_final_check(
                training_input, input_size, "training input final check"
            )
            NetworkTools.general_final_check(
                training_output, output_size, "training output final check"
            )
            NetworkTools.general_final_check(
                evaluation_input, input_size, "evaluation input final check"
            )
            NetworkTools.general_final_check(
                evaluation_output, output_size, "evaluation output final check"
            )

            # define network
            self.net = Network2.Network(
                [input_size, output_size], cost = Network2.CrossEntropyCost
            )
        
            # train
            (
                evaluation_cost,
                evaluation_accuracy,
                training_cost,
                training_accuracy,
            ) = self.net.SGD(
                zip(training_input, training_output),
                20,  # max epochs
                1,  # batch size
                2.0,  # learning rate
                0.1, # lambda
                evaluation_data = zip(evaluation_input, evaluation_output),
                eta_change_interval = 5, # learning rate change interval
                eta_change_factor = 0.5 # learning rate change factor
            )

            # Postprocessing
            NetworkTools.draw_stats(
                training_cost,
                training_accuracy,
                evaluation_cost,
                evaluation_accuracy,
                predicted_phase,
            )
            print("Phase " + predicted_phase + " training completed.")
    
    def get_results_debug(self, phases, games):
        # get size
        input_size = len(games[0].team1.get_stats()) * 2
        output_size = 80

        predictions = []
        for game in games:
            if game.phase == phases[1]:
                input_data = numpy.array(game.get_input()).reshape(input_size, 1)
                game.predicted_score = ScoreClass.Score(self.net.feedforward(input_data))
                predictions.append(game.get_csv_format() + [item[0] for item in self.net.feedforward(input_data)])

        # output to excel
        # columns = ['Phase', 'Team1', 'Team2', 'T1_r1', 'T1_final', 'T1_extra', 'T1_penalties', 'T2_r1', 'T2_final', 'T2_extra', 'T2_penalties']
        # df = pd.DataFrame([game.get_csv_format() for game in games])
        df = pd.DataFrame(predictions, columns = GlobalParameters.get_columns())
        df.to_csv("Output.csv", index = False)

    def get_results(self, phases, games):
        # get size
        input_size = len(games[0].team1.get_stats()) * 2
        output_size = 80

        for game in games:
            if game.is_unplayed:
                input_data = numpy.array(game.get_input()).reshape(input_size, 1)
                game.predicted_result = self.net.feedforward(input_data)

        # output to excel
        columns = ['Phase', 'Team1', 'Team2', 'T1_r1', 'T1_final', 'T1_extra', 'T1_penalties', 'T2_r1', 'T2_final', 'T2_extra', 'T2_penalties']
        df = pd.DataFrame([game.get_csv_format() for game in games], columns = columns)
        df.to_csv("Output.csv", index = False)
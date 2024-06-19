import numpy
import GeneralPredictor


'''
Each team has its own network to predict the next game.
Initially designed so that parameters of the initial networks were to be used as stats in each game.
A final network would then train over all games.
'''
class PredictByTeam(GeneralPredictor.GeneralPredictor):
    def _get_network_input_size(self):
        return 108
    
    def _get_network_output_size(self):
        return 80

    def get_game_data(self, game):
        # input
        input_data = game.get_input()
        input_data = numpy.array(input_data).reshape((len(input_data), 1))
        self._general_check(input_data, self._get_network_input_size(), "input data for game" + game.print())
        # output
        output_data = game.get_output()
        output_data = numpy.array(output_data).reshape((len(output_data), 1))
        self._general_check(output_data, self._get_network_output_size(), "output data for game" + game.print())
        return (input_data, output_data)

    def train(self, phases, teams, games):
        for predicted_phase in phases[1:]:
            if predicted_phase > phases[1]:
                break

            print("Phase " + predicted_phase + " training started.")
            nr_remaining_teams = 0

            for team in teams:
                if team.plays_phase(predicted_phase):
                    self._reset_total_cost_accuracy()
                    nr_remaining_teams += 1
                    training_input = []
                    training_output = []
                    evaluation_input = []
                    evaluation_output = []
                    for game in team.team_games:
                        # training data
                        if game.phase < predicted_phase:
                            input_data, output_data = self.get_game_data(game)
                            training_input.append(input_data)
                            training_output.append(output_data)
                        # evaluation data
                        if game.phase == predicted_phase:
                            input_data, output_data = self.get_game_data(game)
                            evaluation_input.append(input_data)
                            evaluation_output.append(output_data)
                    self._training_wrapper(training_input, training_output, evaluation_input, evaluation_output)

            print("Phase " + predicted_phase + " training completed.")

            self.total_training_cost = [x / nr_remaining_teams for x in self.total_training_cost]
            self.total_training_accuracy = [x / nr_remaining_teams * 100 for x in self.total_training_accuracy]
            self.total_evaluation_cost = [x / nr_remaining_teams for x in self.total_evaluation_cost]
            self.total_evaluation_accuracy = [x / nr_remaining_teams * 100 for x in self.total_evaluation_accuracy]
            self._draw_stats(predicted_phase)

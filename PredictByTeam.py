import numpy
import GeneralPredictor


'''
Each team has its own network to predict the next game.
Initially designed so that parameters of the initial networks were to be used as stats in each game.
A final network would then train over all games.
'''
class PredictByTeam(GeneralPredictor.GeneralPredictor):
    def get_network_input_size(self):
        return 108
    
    def get_network_output_size(self):
        return 80

    def get_game_data(self, game):
        # input
        input_data = game.team1_stats + game.team2_stats
        input_data = numpy.array(input_data).reshape((len(input_data), 1))
        assert input_data.shape[0] == self.get_network_input_size(), "Invalid shape %d, expected %d, for get_game_input_data for game %s" % (
            input_data.shape[0],
            self.get_network_input_size(),
            game.print(),
        )
        assert input_data.shape[1] == 1, "Invalid shape %d, expected %d, for get_training_data for game %s" % (
            input_data.shape[1],
            1,
            game.print(),
        )
        # output
        output_data = game.team1_output + game.team2_output
        output_data = numpy.array(output_data).reshape((len(output_data), 1))
        assert output_data.shape[0] == self.get_network_output_size(), "Invalid shape %d, expected %d, for get_game_output_data of game %s" % (
            output_data.shape[0],
            self.get_network_output_size(),
            game.print(),
        )
        assert output_data.shape[1] == 1, "Invalid shape %d, expected %d, for GenerateOutput of game %s" % (
            output_data.shape[1],
            1,
            game.print(),
        )
        return (input_data, output_data)

    def get_training_data(self, team, phase):
        training_input = []
        training_output = []
        for game in team.team_games:
            if game.phase < phase:
                input_data, output_data = self.get_game_data(game)
                training_input.append(input_data)
                training_output.append(output_data)
        return (training_input, training_output)

    def get_evaluation_data(self, team, phase):
        evaluation_input = []
        evaluation_output = []
        for game in team.team_games:
            if game.phase == phase:
                input_data, output_data = self.get_game_data(game)
                evaluation_input.append(input_data)
                evaluation_output.append(output_data)
        return (evaluation_input, evaluation_output)

    def get_results(self, phases, teams):
        for predicted_phase in phases[1:]:
            print("Phase " + predicted_phase + " training started.")
            nr_remaining_teams = 0

            for team in teams:
                if team.plays_phase(predicted_phase):
                    nr_remaining_teams += 1
                    (training_input, training_output) = self.get_training_data(team, predicted_phase)
                    (evaluation_input, evaluation_output) = self.get_evaluation_data(team, predicted_phase)
                    self.network_wrapper(training_input, training_output, evaluation_input, evaluation_output)

            self.total_training_cost = [x / nr_remaining_teams for x in self.total_training_cost]
            self.total_training_accuracy = [x / nr_remaining_teams for x in self.total_training_accuracy]
            self.total_evaluation_cost = [x / nr_remaining_teams for x in self.total_evaluation_cost]
            self.total_evaluation_accuracy = [x / nr_remaining_teams for x in self.total_evaluation_accuracy]
            self.draw_stats(predicted_phase)
            print("Phase " + predicted_phase + " training completed.")

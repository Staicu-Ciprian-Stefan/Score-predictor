import numpy
import statistics
import GeneralPredictor


class PredictByGame(GeneralPredictor.GeneralPredictor):
    def _get_network_input_size(self):
        return 376
    
    def _get_network_output_size(self):
        return 80

    def train(self, phases, teams, games):
        for predicted_phase in phases[1:]:
            print("Phase " + predicted_phase + " training started.")
            self._reset_total_cost_accuracy()
            training_nr_games = 0
            evaluation_nr_games = 0

            training_input = []
            training_output = []
            evaluation_input = []
            evaluation_output = []
            for game in games:
                t1 = game.team1.get_stats(predicted_phase)
                t2 = game.team2.get_stats(predicted_phase)
                input = numpy.vstack(t1 + t2)
                output = numpy.array(game.get_output()).reshape((self._get_network_output_size(), 1))
                # training data
                if game.phase < predicted_phase:
                    training_nr_games += 1
                    training_input.append(input)
                    training_output.append(output)
                # evaluation data
                if game.phase == predicted_phase:
                    evaluation_nr_games += 1
                    evaluation_input.append(input)
                    evaluation_output.append(output)
            
            self._training_wrapper(training_input, training_output, evaluation_input, evaluation_output)
            print("Phase " + predicted_phase + " training completed.")
            
            self.total_training_cost = [x / training_nr_games for x in self.total_training_cost]
            self.total_training_accuracy = [x / training_nr_games * 100 for x in self.total_training_accuracy]
            self.total_evaluation_cost = [x / evaluation_nr_games for x in self.total_evaluation_cost]
            self.total_evaluation_accuracy = [x / evaluation_nr_games * 100 for x in self.total_evaluation_accuracy]
            self._draw_stats(predicted_phase)
            print("Phase " + predicted_phase + " training completed.")
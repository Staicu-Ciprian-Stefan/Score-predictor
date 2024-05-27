'''
import GeneralPredictions

def generate_input_v2(self, predicted_phase):
    result = self.team1.GetTeamStatistics(predicted_phase) + self.team2.GetTeamStatistics(predicted_phase)
    return numpy.array(result)

class PredictByGame(GeneralPredictions.GeneralPredictor):
    def get_network_parameters():
        return -1

    def get_inputs_outputs():
        training_nr_games = 0
        evaluation_nr_games = 0
        for game in games:
            if game.Phase < predicted_phase:
                training_nr_games += 1
                training_input.append(game.GenerateInput2(predicted_phase))
                training_output.append(game.GenerateOutput())
            else:
                evaluation_nr_games += 1
                evaluation_input.append(game.GenerateInput2(predicted_phase))
                evaluation_output.append(game.GenerateOutput())
        assert training_input[0].shape[0] == 108, "Invalid shape %d, expected %d" % (
            training_input[0].shape[0],
            108,
        )
        assert training_input[0].shape[1] == 1, "Invalid shape %d, expected %d" % (
            training_input[0].shape[1],
            1,
        )
        assert training_output[0].shape[0] == 80, "Invalid shape %d, expected %d" % (
            training_output[0].shape[0],
            80,
        ) # ok
        assert training_output[0].shape[1] == 1, "Invalid shape %d, expected %d" % (
            training_output[0].shape[1],
            1,
        ) # ok
        assert evaluation_input[0].shape[0] == 108, "Invalid shape %d, expected %d" % (
            evaluation_input[0].shape[0], 
            108
        )
        assert evaluation_input[0].shape[1] == 1, "Invalid shape %d, expected %d" % (
            evaluation_input[0].shape[1],
            1,
        )
        assert evaluation_output[0].shape[0] == 80, "Invalid shape %d, expected %d" % (
            evaluation_output[0].shape[0], 
            80
        ) # ok
        assert evaluation_output[0].shape[1] == 1, "Invalid shape %d, expected %d" % (
            evaluation_output[0].shape[1], 
            1
        ) # ok
    
    def get_results(self, phases, games):
        for predicted_phase in phases[1:]:
            print("Phase " + predicted_phase + " training started.")
            self.training_cost = []
            self.training_accuracy = []
            self.evaluation_cost = []
            self.evaluation_accuracy = []
            net = Network2.Network([108, 80], cost=Network2.CrossEntropyCost)
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
                evaluation_data=zip(evaluation_input, evaluation_output),
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_cost=True,
                monitor_training_accuracy=True,
            )
            if TotalTrainingCost == []:
                TotalTrainingCost = training_cost
            else:
                TotalTrainingCost = [
                    sum(x) for x in zip(TotalTrainingCost, training_cost)
                ]
            if TotalTrainingAccuracy == []:
                TotalTrainingAccuracy = training_accuracy
            else:
                TotalTrainingAccuracy = [
                    sum(x) for x in zip(TotalTrainingAccuracy, training_accuracy)
                ]
            if TotalEvaluationCost == []:
                TotalEvaluationCost = evaluation_cost
            else:
                TotalEvaluationCost = [
                    sum(x) for x in zip(TotalEvaluationCost, evaluation_cost)
                ]
            if TotalEvaluationAccuracy == []:
                TotalEvaluationAccuracy = evaluation_accuracy
            else:
                TotalEvaluationAccuracy = [
                    sum(x) for x in zip(TotalEvaluationAccuracy, evaluation_accuracy)
                ]

            draw_stats(
                predicted_phase,
                TotalTrainingCost,
                [x / training_nr_games for x in TotalTrainingAccuracy],
                TotalEvaluationCost,
                [x / evaluation_nr_games for x in TotalEvaluationAccuracy],
            )
            print("Phase " + predicted_phase + " training completed.")
'''
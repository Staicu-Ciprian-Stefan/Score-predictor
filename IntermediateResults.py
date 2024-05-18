import Network2

import matplotlib.pyplot as plt
import pandas as pd

def DrawStats(Phase, trainingCost, trainingAccuracy, evaluationCost, evaluationAccuracy):
    fig = plt.figure()
    fig.suptitle(Phase + ' stats', fontsize=16)
    tc = fig.add_subplot(221)
    tc.plot(trainingCost)
    tc.set_ylim(bottom=0)
    tc.set_xlim(left=0)
    tc.grid(True)
    # tc.set_xlabel("Epoch")
    tc.set_ylabel("Cost")
    tc.set_title("Training Cost")

    ta = fig.add_subplot(222)
    ta.plot(trainingAccuracy)
    ta.set_ylim(bottom=0, top=1)
    ta.set_xlim(left=0)
    ta.grid(True)
    # tc.set_xlabel("Epoch")
    ta.set_ylabel("Accuracy %")
    ta.set_title("Training Accuracy")

    ec = fig.add_subplot(223)
    ec.plot(evaluationCost)
    ec.set_ylim(bottom=0)
    ec.set_xlim(left=0)
    ec.grid(True)
    ec.set_xlabel("Epoch")
    ec.set_ylabel("Cost")
    ec.set_title("Evaluation Cost")

    ea = fig.add_subplot(224)
    ea.plot(evaluationAccuracy)
    ea.set_ylim(bottom=0, top=1)
    ea.set_xlim(left=0)
    ea.grid(True)
    ea.set_xlabel("Epoch")
    ea.set_ylabel("Accuracy %")
    ea.set_title("Evaluation Accuracy")

    plt.show()

def PrintStats(Phases, Teams, trainingCost, trainingAccuracy, evaluationCost, evaluationAccuracy):
    x = 0

def GetIntermediateResults(Phases, Teams):
    for phaseToPredict in Phases[1:]:
        print("Phase " + phaseToPredict + " training started.")
        TotalEvaluationCost = []
        TotalEvaluationAccuracy = []
        TotalTrainingCost = []
        TotalTrainingAccuracy = []
        for team in Teams:
            if team.PlaysPhase(phaseToPredict):
                # print("Phase " + phaseToPredict + " , Team " + team.Name + " training started.")
                trainingInput, trainingOutput = team.TrainingData(phaseToPredict)
                evaluationInput, evaluationOutput = team.EvaluationData(phaseToPredict)
                assert trainingInput[0].shape[0] == 108, "Invalid shape %d, expected %d" % (
                    trainingInput[0].shape[0],
                    108,
                )
                assert trainingInput[0].shape[1] == 1, "Invalid shape %d, expected %d" % (
                    trainingInput[0].shape[1],
                    1,
                )
                assert trainingOutput[0].shape[0] == 80, "Invalid shape %d, expected %d" % (
                    trainingOutput[0].shape[0],
                    80,
                )
                assert trainingOutput[0].shape[1] == 1, "Invalid shape %d, expected %d" % (
                    trainingOutput[0].shape[1],
                    1,
                )
                assert (
                    evaluationInput[0].shape[0] == 108
                ), "Invalid shape %d, expected %d" % (evaluationInput[0].shape[0], 108)
                assert evaluationInput[0].shape[1] == 1, "Invalid shape %d, expected %d" % (
                    evaluationInput[0].shape[1],
                    1,
                )
                assert (
                    evaluationOutput[0].shape[0] == 80
                ), "Invalid shape %d, expected %d" % (evaluationOutput[0].shape[0], 80)
                assert (
                    evaluationOutput[0].shape[1] == 1
                ), "Invalid shape %d, expected %d" % (evaluationOutput[0].shape[1], 1)
                net = Network2.Network([108, 80], cost=Network2.CrossEntropyCost)
                (
                    evaluation_cost,
                    evaluation_accuracy,
                    training_cost,
                    training_accuracy,
                ) = net.SGD(
                    zip(trainingInput, trainingOutput),
                    10,  # epochs
                    1,  # batch size
                    2.0,  # learning rate
                    evaluation_data=zip(evaluationInput, evaluationOutput),
                    monitor_evaluation_cost=True,
                    monitor_evaluation_accuracy=True,
                    monitor_training_cost=True,
                    monitor_training_accuracy=True,
                )
                # print("Phase " + phaseToPredict + " , Team " + team.Name + " training complete.")
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
        DrawStats(
            phaseToPredict,
            TotalTrainingCost,
            [x / 32 for x in TotalTrainingAccuracy],
            TotalEvaluationCost,
            [x / 32 for x in TotalEvaluationAccuracy],
        )
        print("Phase " + phaseToPredict + " training completed.")
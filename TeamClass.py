
def SeparateGamesByTeam(TeamNames, Games):
    Teams = []
    for team in TeamNames:
        TeamGames = []
        for game in filter(lambda x: x.Team1 == team or x.Team2 == team, Games):
            TeamGames.append(game)
            if TeamGames[-1].Team2 == team:
                TeamGames[-1].ReverseTeams()
        Teams.append(Team(team, TeamGames))
    return Teams


class Team:
    def __init__(self, name, teamGames):
        self.Name = name
        self.TeamGames = teamGames
        self.TrainedParameters = []

    def TrainingData(self, phase):
        trainingInput = []
        trainingOutput = []
        for game in self.TeamGames:
            if game.Phase < phase:
                trainingInput.append(game.GenerateInput())
                trainingOutput.append(game.GenerateOutput())
        return (trainingInput, trainingOutput)

    def EvaluationData(self, phase):
        evaluationInput = []
        evaluationOutput = []
        for game in self.TeamGames:
            if game.Phase == phase:
                evaluationInput.append(game.GenerateInput())
                evaluationOutput.append(game.GenerateOutput())
        return (evaluationInput, evaluationOutput)

    def PrintScores(self):
        for game in self.TeamGames:
            game.PrintScore()

    def PlaysPhase(self, phase):
        for game in self.TeamGames:
            if phase <= game.Phase:
                return True
        return False
    
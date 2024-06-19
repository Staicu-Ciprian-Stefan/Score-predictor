import numpy
import statistics

class Team:
    def __init__(self, name, team_games):
        self.name = name
        self.team_games = team_games
    
    def __str__(self):
        return f'{self.name} xxx'

    def plays_phase(self, phase):
        for game in self.team_games:
            if phase <= game.phase:
                return True
        return False
    
    def get_stats(self, phase):
        # get all the stats
        values = []
        for game in self.team_games:
            if game.phase < phase:
                values.append(game.get_input() + game.get_output())
        # transpose stats
        transposed = list(zip(*values))
        # compute averages and standard deviations
        averages = [sum(column) / len(column) for column in transposed]

        if len(values) > 1:
            standard_deviations = [statistics.stdev(column) for column in transposed]
        else:
            standard_deviations = [0] * len(transposed)

        # format result as numpy array
        result = numpy.array(averages + standard_deviations)
        return result
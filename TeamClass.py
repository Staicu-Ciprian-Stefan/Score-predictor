import numpy
import statistics


class Team:
    def __init__(self, name, team_games):
        self.name = name
        self.team_games = team_games

    def plays_phase(self, phase):
        for game in self.team_games:
            if phase <= game.phase:
                return True
        return False
    
    def get_team_stats(self, phase):
        # get all the stats
        values = []
        for game in self.team_games:
            if game.phase < phase:
                values.append(numpy.vstack(game.generate_input() + game.generate_output()))
        # transpose stats
        transposed = list(zip(*values))
        # compute averages
        averages = [sum(column) / len(column) for column in transposed]
        standard_deviations = [statistics.stdev(column) for column in transposed]

        return (averages, standard_deviations)
    
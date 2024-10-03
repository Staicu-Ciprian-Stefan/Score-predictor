# standard libraries
import math
import pandas

# 3rd party libraries

# my libraries
import ScoreClass

class Game:
    # format must be kept with team12, score12, stats12 for reverse purposes
    def get_stats(self):
        return self.team1_stats + self.team2_stats + self.score.get_output()
        
    def get_input(self, phase = None):
        return self.team1.get_stats(phase) + self.team1.get_stats(phase)
    
    def get_output(self):
        return self.score.get_output()

    def get_csv_format(self):
        return [self.phase, self.team1_name, self.team2_name]
    
    def __init__(self, raw_data_line):
        self.team1 = -1
        self.team2 = -1
        self.phase = raw_data_line[0]
        self.team1_name = raw_data_line[1]
        self.team2_name = raw_data_line[2]
        
        # self.team1_score = raw_data_line[3:10:2]
        # self.team2_score = raw_data_line[4:11:2]

        self.team1_stats = raw_data_line[11::2]
        self.team2_stats = raw_data_line[12::2]

        # determine the game type
        self.is_unknown = pandas.isna(self.team1_name) or pandas.isna(self.team2_name)
        self.is_unplayed = any(pandas.isna(x) for x in self.team1_stats + self.team2_stats)
        self.is_played = not(self.is_unknown or self.is_unplayed)

        self.score = -1
        self.predicted_score = None
        if self.is_played:
            self.score = ScoreClass.Score(raw_data_line[3:11])

    def add_team_reference(self, teams):
        if not self.is_unknown:
            self.team1 = list(filter(lambda x: x.name == self.team1_name, teams))[0]
            self.team2 = list(filter(lambda x: x.name == self.team2_name, teams))[0]

    def reverse_teams(self):
        if self.is_unknown:
            raise NotImplementedError
        
        if self.is_unplayed:
            placeholder = self.team1_name
            self.team1_name = self.team2_name
            self.team2_name = placeholder
            placeholder = self.team1
            self.team1 = self.team2
            self.team2 = placeholder
            
        if self.is_played:
            placeholder = self.team1_name
            self.team1_name = self.team2_name
            self.team2_name = placeholder

            placeholder = self.team1_stats
            self.team1_stats = self.team2_stats
            self.team2_stats = placeholder

            placeholder = self.team1
            self.team1 = self.team2
            self.team2 = placeholder

            self.score.reverse_teams()
            
    def print(self):
        return "(" + self.phase + " " + self.team1_name + "-" + self.team2_name + ")"
# standard libraries
import math
import pandas

# 3rd party libraries

# my libraries
import NetworkTools

class Game:
    # format must be kept with team12, score12, stats12 for reverse purposes
    def get_stats(self):
        return self.team1_stats + self.team2_stats + self.team1_output + self.team2_output
        
    def get_input(self, phase = None):
        return self.team1.get_stats(phase) + self.team1.get_stats(phase)
    
    def get_output(self):
        return self.team1_output + self.team2_output

    def get_csv_format(self):
        return [self.phase, self.team1_name, self.team2_name]
    
    def __init__(self, raw_data_line):
        self.team1 = -1
        self.team2 = -1
        self.phase = raw_data_line[0]
        self.team1_name = raw_data_line[1]
        self.team2_name = raw_data_line[2]
        
        self.team1_score = raw_data_line[3:10:2]
        self.team2_score = raw_data_line[4:11:2]

        self.team1_stats = raw_data_line[11::2]
        self.team2_stats = raw_data_line[12::2]

        self.predicted_result = None

        # determine the game type
        self.is_unknown = pandas.isna(self.team1_name) or pandas.isna(self.team2_name)
        self.is_unplayed = any(pandas.isna(x) for x in self.team1_stats + self.team2_stats)
        self.is_played = not(self.is_unknown or self.is_unplayed)

        if self.is_played:
            # vectorize results
            self.team1_output = NetworkTools.vectorized_result(raw_data_line[3])
            self.team2_output = NetworkTools.vectorized_result(raw_data_line[4])
            self.team1_output.extend(NetworkTools.vectorized_result(raw_data_line[5]))
            self.team2_output.extend(NetworkTools.vectorized_result(raw_data_line[6]))
            self.team1_output.extend(NetworkTools.vectorized_result(raw_data_line[7]))
            self.team2_output.extend(NetworkTools.vectorized_result(raw_data_line[8]))
            self.team1_output.extend(NetworkTools.vectorized_result(raw_data_line[9]))
            self.team2_output.extend(NetworkTools.vectorized_result(raw_data_line[10]))

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
            placeholder = self.team1_score
            self.team1_score = self.team2_score
            self.team2_score = placeholder
            placeholder = self.team1_output
            self.team1_output = self.team2_output
            self.team2_output = placeholder
            placeholder = self.team1
            self.team1 = self.team2
            self.team2 = placeholder
            
    def print(self):
        return "(" + self.phase + " " + self.team1_name + "-" + self.team2_name + ")"
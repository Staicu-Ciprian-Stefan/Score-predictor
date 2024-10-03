import NetworkTools

class Score:
    def __init__(self, input_score):
        # first case is for initialization, second is for feedforward result
        if len(input_score) == 8:
            # initial parameters
            self.team1_score = TeamScore(input_score[::2])
            self.team2_score = TeamScore(input_score[1::2])

            if self.team1_score.penalties + self.team2_score.penalties > 0:
                self.is_penalties = 1
            else:
                self.is_penalties = 0
            if self.team1_score.final_120 + self.team1_score.final_120 > 0:
                self.is_extra = 1
            else:
                self.is_extra = 0
        else:
            self.is_extra = input_score[0]
            self.is_penalties = input_score[1]

            t1_half1 = max(input_score[2:12])
            t1_half2 = max(input_score[12:22])
            t1_final_90 =  max(input_score[22:32]) 
            t1_extra = max(input_score[32:42]) 
            t1_final_120 = max(input_score[42:52]) 
            t1_penalties = max(input_score[52:62])

            t2_half1 = max(input_score[62:72])
            t2_half2 = max(input_score[72:82])
            t2_final_90 =  max(input_score[82:92]) 
            t2_extra = max(input_score[92:102]) 
            t2_final_120 = max(input_score[102:112]) 
            t2_penalties = max(input_score[112:122])

            self.team1_score = TeamScore([t1_half1, t1_final_90, t1_final_120, t1_penalties])
            self.team1_score = TeamScore([t2_half1, t2_final_90, t2_final_120, t2_penalties])

    # size = 122
    def get_output(self):
        return [self.is_extra, self.is_penalties] + self.team1_score.output + self.team2_score.output
    
    def get_short_output(self):
        return [self.is_extra, self.is_penalties] + self.team1_score.get_short_output() + self.team2_score.get_short_output()

    def reverse_teams(self):
        placeholder = self.team1_score
        self.team1_score = self.team2_score
        self.team2_score = placeholder

class TeamScore:
    def __init__(self, input_score):
        # initial parameters
        self.half1 = input_score[0]
        self.final_90 = input_score[1]
        self.final_120 = input_score[2]
        self.penalties = input_score[3]

        # extra parameters
        self.half2 = self.final_90 - self.half1
        self.extra = self.final_120 - self.final_90

        # precompute output
        self.output = NetworkTools.vectorized_result(self.half1)
        self.output.extend(NetworkTools.vectorized_result(self.half2))
        self.output.extend(NetworkTools.vectorized_result(self.final_90))
        self.output.extend(NetworkTools.vectorized_result(self.extra))
        self.output.extend(NetworkTools.vectorized_result(self.final_120))
        self.output.extend(NetworkTools.vectorized_result(self.penalties))

    def get_short_output(self):
        return [self.half1, self.half2, self.final_90, self.extra, self.final_120, self.penalties]
import VectorizationTools

class Game:
    # format must be kept with team12, score12, stats12 for reverse purposes
    def __init__(self, half1, half2, extra, penalties):
        self.team1 = -1
        self.team2 = -1
        self.phase = half1[0]
        self.team1_name = half1[1]
        self.team2_name = half1[2]
        # 3 is the comment signaling if it is half1, half2, extra or penalties
        self.team1_score = [
            half1[4],
            half2[4],
            extra[4],
            penalties[4],
        ]
        self.team2_score = [
            half1[5],
            half2[5],
            extra[5],
            penalties[5],
        ]

        self.team1_stats = half1[6::2]
        self.team2_stats = half1[7::2]
        self.team1_stats.extend(half2[6::2])
        self.team2_stats.extend(half2[7::2])
        self.team1_stats.extend(extra[6::2])
        self.team2_stats.extend(extra[7::2])

        # on penalties score is the only info

        self.team1_output = VectorizationTools.VectorizationMethod(half1[4])
        self.team2_output = VectorizationTools.VectorizationMethod(half1[5])
        self.team1_output.extend(VectorizationTools.VectorizationMethod(half2[4]))
        self.team2_output.extend(VectorizationTools.VectorizationMethod(half2[5]))
        self.team1_output.extend(VectorizationTools.VectorizationMethod(extra[4]))
        self.team2_output.extend(VectorizationTools.VectorizationMethod(extra[5]))
        self.team1_output.extend(VectorizationTools.VectorizationMethod(penalties[4]))
        self.team2_output.extend(VectorizationTools.VectorizationMethod(penalties[5]))

    def add_team_reference(self, teams):
        self.team1 = list(filter(lambda x: x.name == self.team1_name, teams))[0]
        self.team2 = list(filter(lambda x: x.name == self.team2_name, teams))[0]

    def reverse_teams(self):
        placeholder = self.team1_name
        self.team1_name = self.team2_name
        self.team2_name = placeholder
        placeholder = self.team1_stats
        self.team1_stats = self.team2_stats
        self.team2_stats = placeholder
        placeholder = self.team1_score
        self.team1_score = self.team2_score
        self.team2_score = placeholder

    def print(self):
        print(self.phase, " ", self.team1_name, " - ", self.team2_name)
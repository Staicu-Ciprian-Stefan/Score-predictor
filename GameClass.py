import numpy
import VectorizationTools

class Game:
    # format must be kept with team12, score12, stats12 for reverse purposes
    def __init__(self, half1, half2, extra, penalties):
        self.Phase = half1[0]
        self.Team1 = half1[1]
        self.Team2 = half1[2]
        # 3 is the comment signaling if it is half1, half2, extra or penalties
        self.Score1 = [
            half1[4],
            half2[4],
            extra[4],
            penalties[4],
        ]
        self.Score2 = [
            half1[5],
            half2[5],
            extra[5],
            penalties[5],
        ]

        self.Stats1 = half1[6::2]
        self.Stats2 = half1[7::2]
        self.Stats1.extend(half2[6::2])
        self.Stats2.extend(half2[7::2])
        self.Stats1.extend(extra[6::2])
        self.Stats2.extend(extra[7::2])

        # on penalties score is the only info

        self.Output1 = VectorizationTools.VectorizationMethod(half1[4])
        self.Output2 = VectorizationTools.VectorizationMethod(half1[5])
        self.Output1.extend(VectorizationTools.VectorizationMethod(half2[4]))
        self.Output2.extend(VectorizationTools.VectorizationMethod(half2[5]))
        self.Output1.extend(VectorizationTools.VectorizationMethod(extra[4]))
        self.Output2.extend(VectorizationTools.VectorizationMethod(extra[5]))
        self.Output1.extend(VectorizationTools.VectorizationMethod(penalties[4]))
        self.Output2.extend(VectorizationTools.VectorizationMethod(penalties[5]))

    def ReverseTeams(self):
        placeholder = self.Team1
        self.Team1 = self.Team2
        self.Team2 = placeholder
        placeholder = self.Stats1
        self.Stats1 = self.Stats2
        self.Stats2 = placeholder
        placeholder = self.Score1
        self.Score1 = self.Score2
        self.Score2 = placeholder

    def GenerateInput(self):
        # no need to assert stats since they were already verified on initial check

        result = self.Stats1 + self.Stats2
        result = numpy.array(result).reshape((len(result), 1))
        
        assert result.shape[0] == 2 * len(self.Stats1), "Invalid shape %d, expected %d, for GenerateInput of game %s" % (
            result.shape[0],
            2 * len(self.Stats1),
            self.Phase + self.Team1 + self.Team2,
        )
        assert result.shape[1] == 1, "Invalid shape %d, expected %d, for GenerateInput of game %s" % (
            result.shape[1],
            1,
            self.Phase + self.Team1 + self.Team2,
        )
        return result

    def GenerateOutput(self):
        result = self.Output1 + self.Output2
        result = numpy.array(result).reshape((len(result), 1))
        assert result.shape[0] == 2 * len(self.Output1), "Invalid shape %d, expected %d, for GenerateOutput of game %s" % (
            result.shape[0],
            2 * len(self.Output1),
        )
        assert result.shape[1] == 1, "Invalid shape %d, expected %d, for GenerateOutput of game %s" % (
            result.shape[1],
            1,
            self.Phase + self.Team1 + self.Team2,
        )
        return result

    def PrintNames(self):
        print(self.Phase, self.Team1, self.Team2)

    def PrintScore(self):
        self.PrintNames()
        print(self.Score1, " ", self.Score2)

    def PrintData(self):
        self.PrintScore()
        print(numpy.around(self.Stats1, decimals=2))
        print(numpy.around(self.Stats2, decimals=2))

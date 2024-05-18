import DataLoader
import TeamClass
import IntermediateResults

def main():
    Games, TeamNames, Phases = DataLoader.ReadData("Fifa2022Scores.xlsx")
    Teams = TeamClass.SeparateGamesByTeam(TeamNames, Games)
    IntermediateResults.GetIntermediateResults(Phases, Teams)

    # print(numpy.around(DataLoader.NormalizedLogisticVectorizedResult(5), decimals=3))

main()
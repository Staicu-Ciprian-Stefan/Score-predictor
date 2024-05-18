import pandas
import GameClass


def ReadData(fileName):
    # read data from excel file, need to use this syntax to read specific sheet
    Fifa2022Scores = pandas.read_excel(fileName, sheet_name=["StatsNormalized"])
    Fifa2022Scores = Fifa2022Scores.get("StatsNormalized").values.tolist()

    # sort stats according to half
    Half1 = Fifa2022Scores[0::4]
    Half2 = Fifa2022Scores[1::4]
    ExtraTime = Fifa2022Scores[2::4]
    Penalties = Fifa2022Scores[3::4]

    # create game list
    Games = []
    for (h1, h2, e, p) in zip(Half1, Half2, ExtraTime, Penalties):
        Games.append(GameClass.Game(h1, h2, e, p))

    # get phase names and team names from game list
    TeamNames = set([])
    Phases = set([])
    for game in Games:
        Phases.add(game.Phase)
        TeamNames.add(game.Team1)
        TeamNames.add(game.Team2)

    TeamNames = list(TeamNames)
    TeamNames.sort()
    Phases = list(Phases)
    Phases.sort()
    return CheckData(Games, TeamNames, Phases)


def CheckData(Games, TeamNames, Phases):
    team_stats_length = len(Games[0].Stats1)
    team_score_length = 40 # this is constant

    for game in Games:
        assert len(game.Stats1) == team_stats_length, "Invalid shape %d, expected %d, for Stats1 of game %s" % (
            len(game.Stats1),
            team_stats_length,
            game.Phase + game.Team1 + game.Team2,
        )
        assert len(game.Stats2) == team_stats_length, "Invalid shape %d, expected %d, for Stats2 of game %s" % (
            len(game.Stats2),
            team_stats_length,
            game.Phase + game.Team1 + game.Team2,
        )
        assert len(game.Output1) == team_score_length, "Invalid shape %d, expected %d, for Stats2 of game %s" % (
            len(game.Output1),
            team_score_length,
            game.Phase + game.Team1 + game.Team2,
        )
        assert len(game.Output2) == team_score_length, "Invalid shape %d, expected %d, for Stats2 of game %s" % (
            len(game.Output2),
            team_score_length,
            game.Phase + game.Team1 + game.Team2,
        )

    return (Games, TeamNames, Phases)
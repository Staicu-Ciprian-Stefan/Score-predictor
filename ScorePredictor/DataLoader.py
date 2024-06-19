import pandas
import GameClass
import TeamClass


def read_data(fileName):
    # read data from excel file, need to use this syntax to read specific sheet
    fifa2022_scores = pandas.read_excel(fileName, sheet_name=["StatsNormalized"])
    fifa2022_scores = fifa2022_scores.get("StatsNormalized").values.tolist()

    # sort stats according to half
    half1 = fifa2022_scores[0::4]
    half2 = fifa2022_scores[1::4]
    extra_time = fifa2022_scores[2::4]
    penalties = fifa2022_scores[3::4]

    # create game list
    games = []
    for (h1, h2, e, p) in zip(half1, half2, extra_time, penalties):
        games.append(GameClass.Game(h1, h2, e, p))

    # get phase names and team names from game list
    team_names = set([])
    phases = set([])
    for game in games:
        phases.add(game.phase)
        team_names.add(game.team1_name)
        team_names.add(game.team2_name)

    team_names = list(team_names)
    team_names.sort()
    phases = list(phases)
    phases.sort()

    # create teams
    teams = []
    for team in team_names:
        team_games = []
        for game in filter(lambda x: x.team1_name == team or x.team2_name == team, games):
            team_games.append(game)
            if team_games[-1].team2_name == team:
                team_games[-1].reverse_teams()
        teams.append(TeamClass.Team(team, team_games))

    # add references
    for game in games:
        game.add_team_reference(teams)

    return check_data(games, teams, team_names, phases)


def check_data(games, teams, team_names, phases):
    team_stats_length = len(games[0].team1_stats)
    team_score_length = 40 # this is constant

    for game in games:
        assert len(game.team1_stats) == team_stats_length, "Invalid shape %d, expected %d, for team1_stats of game %s" % (
            len(game.team1_stats),
            team_stats_length,
            game.print(),
        )
        assert len(game.team2_stats) == team_stats_length, "Invalid shape %d, expected %d, for team2_stats of game %s" % (
            len(game.team2_stats),
            team_stats_length,
            game.print(),
        )
        assert len(game.team1_output) == team_score_length, "Invalid shape %d, expected %d, for team1_output of game %s" % (
            len(game.team1_output),
            team_score_length,
            game.print(),
        )
        assert len(game.team2_output) == team_score_length, "Invalid shape %d, expected %d, for team2_output of game %s" % (
            len(game.team2_output),
            team_score_length,
            game.print(),
        )

    return (games, teams, team_names, phases)

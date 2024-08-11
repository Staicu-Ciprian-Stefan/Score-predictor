# standard libraries
import copy

# 3rd party libraries
import pandas

# my libraries
import GameClass
import TeamClass


def read_data(file_name, sheet_name):
    # read data from excel file, need to use this syntax to read specific sheet
    df = pandas.read_excel(file_name, sheet_name=[sheet_name])
    raw_data = df.get(sheet_name).values.tolist()

    # get phase names and team names
    team_names = set([])
    phases = set([])
    for line in raw_data:
        phases.add(line[0])
        team_names.add(line[1])
        team_names.add(line[2])

    team_names = list(team_names)
    team_names = [name for name in team_names if not pandas.isna(name)] # prune nan in case of incomplete data
    team_names.sort()
    phases = list(phases)
    phases.sort()

    # check for errors
    if len(team_names) % 4 != 0:
        print('Possible error! Number of teams is ', len(team_names))
    if len(phases) != 7:
        print('Possible error! Number of phases is ', len(phases))

    # create game list
    games = []
    for line in raw_data:
        games.append(GameClass.Game(line))

    # normalize data
    # get the highest value for each stat
    nr_parameters = len(games[0].team1_stats)
    max_values = [0] * nr_parameters
    for index in range(nr_parameters):
        max_values[index] = max([game.team1_stats[index] for game in games if game.is_played] + [game.team2_stats[index] for game in games if game.is_played]) 

    # check if the maximum is zero
    for index in range(nr_parameters):
        if max_values[index] == 0:
            max_values[index] = 1

    # normalize the data
    for index in range(nr_parameters):
        for game in games:
            if game.is_played:
                game.team1_stats[index] = game.team1_stats[index] / max_values[index]
                game.team2_stats[index] = game.team2_stats[index] / max_values[index]

    # create teams
    teams = []
    for team in team_names:
        team_games = []
        for game in filter(lambda x: x.team1_name == team or x.team2_name == team, games):
            if not game.is_unknown:
                team_games.append(copy.copy(game))
                if team_games[-1].team2_name == team:
                    team_games[-1].reverse_teams()
        teams.append(TeamClass.Team(team, team_games))

    # add references
    for game in games:
        game.add_team_reference(teams)
    for team in teams:
        for game in team.team_games:
            game.add_team_reference(teams)
             
    # check data for consistency
    team_stats_length = len(games[0].team1_stats)
    team_score_length = 40 # this is constant

    for game in games:
        if game.is_played:
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
    return (phases, team_names, games, teams)
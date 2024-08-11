# standard libraries

# 3rd party libraries
import pandas as pd

# my libraries
import DataLoader
import PredictByTeam
import PredictByGame
import PredictByTournament

def main():
    # parameters
    file_name = 'CountryTournamentHistory.xlsx'
    # file_name = 'ClubTournamentHistory.xlsx'
    # sheet_name = 'Fifa2018' 
    # sheet_name = 'Fifa2022'
    sheet_name = 'Euro2024'

    (phases, team_names, games, teams) = DataLoader.read_data(file_name, sheet_name)
    
    # (input_size, output_size) = (34, 80)
    # (input_size, output_size) = (108, 80)
    # (input_size, output_size) = (158, 80)

    # predictor = PredictByTeam.PredictByTeam(input_size, output_size)
    predictor = PredictByGame.PredictByGame()
    # predictor = PredictByTournament.PredictByTournament()

    predictor.debug_train(phases, teams, games)
    # predictor.train(phases, teams, games)
    predictor.get_results_debug(phases, games)
    # predictor.get_results(phases, games)

main()
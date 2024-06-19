import DataLoader
import PredictByTeam
import PredictByGame
import PredictByTournament

import pandas as pd

def main():
    games, teams, team_names, phases = DataLoader.read_data("Fifa2022Scores.xlsx")
    
    predictor = PredictByTeam.PredictByTeam()
    # predictor = PredictByGame.PredictByGame()
    # predictor = PredictByTournament.PredictByTournament()

    predictor.train(phases, teams, games)
    results = predictor.get_results(teams, games)

    columns = ['Phase', 'Team1', 'Team2', 'T1_r1', 'T1_final', 'T1_extra', 'T1_penalties', 'T2_r1', 'T2_final', 'T2_extra', 'T2_penalties']
    df = pd.DataFrame([game.get_csv_format() for game in games], columns = columns)
    df.to_csv("Output.csv", index = False)


main()
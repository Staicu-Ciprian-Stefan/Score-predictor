import DataLoader
import PredictByTeam
import PredictByGame
import PredictByTournament

def main():
    Games, Teams, TeamNames, Phases = DataLoader.read_data("Fifa2022Scores.xlsx")
    
    predictor = PredictByTeam.PredictByTeam()
    # predictor = PredictByGame.PredictByGame()
    # predictor = PredictByTournament.PredictByTournament()

    predictor.get_results(Phases, Teams)

main()
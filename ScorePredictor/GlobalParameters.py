# global parameters
is_debug = True
vector_size = 10

# use scale_factor = -1.5, -9 <= scale_factor <= 0, -9 is for classic categorization algorithm
scale_factor = -9

# output format
def get_columns():
    columns = ['Phase', 'Team1', 'Team2', 'is_extra', 'is_penalties']
    for team in ['Team1', 'Team2']:
        for round in ['R1', 'R2', '90', 'extra', '120', 'penalties']:
            for goals in range(10):
                columns.append(team + '_' + round + '_' + str(goals))
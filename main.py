import pulp as pl
import pandas as pd

RANKINGS_CSV_FILE_NAME = 'socials_rankings.csv'
MAX_STUDENTS_PER_SESSION = 20
NUMBER_OF_ROTATIONS = 3

def main():
    """
    This script takes a csv file in the current directory that contains as headers: a student identifier (eg. name, id), names of sessions, and
    data which are integer rankings of the named sessions. The input file name, enrolment cap per session, and number of rotations can be adjusted.
    Note: A student is assumed to be required to enrol in exactly one session per rotation, and may not enrol in the same session in multiple rotations.
    """

    rankings = pd.read_csv(RANKINGS_CSV_FILE_NAME)

    num_students, num_sessions = rankings.shape[0], rankings.shape[1] - 1 # first column is student identifer, not session
    sessions = rankings.columns[1:]
                          
    # Initialize the optimization model
    problem = pl.LpProblem("FlexDayAssignment", pl.LpMinimize)

    # Define assignment matrix
    # X[i, j, k] = 1 if student i is assigned to session j in rotation k, 0 otherwise
    X = pl.LpVariable.dicts("X", ((i, j, k) for i in range(num_students) for j in range(num_sessions) for k in range(NUMBER_OF_ROTATIONS)), cat=pl.LpBinary)
    
    # Specify contsraints

    # Each student attends exactly one session per rotation
    for i in range(num_students):
        # all sessions available in all but last rotation
        for k in range(NUMBER_OF_ROTATIONS): 
            problem += pl.lpSum(X[i, j, k] for j in range(num_sessions)) == 1, f"one_session_per_student_{i}_rotation_{k}"

    # Each student may attend the same session no more than once over all conferences
    for i in range(num_students):
        for j in range(num_sessions):
            problem += pl.lpSum(X[i, j, k] for k in range(NUMBER_OF_ROTATIONS)) <= 1, f"session_{j}_max_once_per_student_{i}"
    # Enrollment cap per session for each rotation
    for j in range(num_sessions):
        for k in range(NUMBER_OF_ROTATIONS):
            problem += pl.lpSum(X[i, j, k] for i in range(num_students)) <= MAX_STUDENTS_PER_SESSION, f"cap_per_session_{j}_rotation_{k}"
    
    # Create list of lists of int rankings from input data
    R = rankings.iloc[:, 1:].to_numpy().tolist()

    # Set the objective
    objective = pl.lpSum((R[i][j] - 1) * X[i, j, k] for i in range(num_students) for j in range(num_sessions) for k in range(NUMBER_OF_ROTATIONS))
    problem += objective
    
    problem.solve()

    # cataloguing for output
    session_enrolments = {k: {j: [] for j in range(num_sessions)} for k in range(NUMBER_OF_ROTATIONS)}
    # for testing
    rankings_of_assigned_sessions_by_student = {i: [] for i in range(num_students)}

    # parse solution variables
    for v in problem.variables():
        i, j, k = map(lambda x: int(x.replace('_', '')), v.name.split('(')[1].split(')')[0].split(','))
        x_ijk = v.value()
        if int(x_ijk) == 1:
            session_enrolments[k][j].append(rankings.iloc[i,0])
            rankings_of_assigned_sessions_by_student[i].append(R[i][j])


    # append nones so we can make the output dataframe
    max_length = max(len(lst) for lst in [session_enrolments[k][j] for j in range(num_sessions) for k in range(NUMBER_OF_ROTATIONS)])

    for k in session_enrolments:
        for j in session_enrolments[k]:
            session_enrolments[k][j] += [None] * (max_length - len(session_enrolments[k][j]))

    # create an output csv file for each rotation
    for k in range(NUMBER_OF_ROTATIONS):
        pd.DataFrame.from_dict(session_enrolments[k]).set_axis(sessions, axis='columns').to_csv(f"enrolments_for_rotation_{k}.csv", index=False)

    # check to see what students ranked their assigned sessions (measure for how well we did)
    for i, vals in rankings_of_assigned_sessions_by_student.items():
        print(rankings.iloc[i,0] + ":", vals)


if __name__ == '__main__':
    main()

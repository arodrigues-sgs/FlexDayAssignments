import pulp as pl
import pandas as pd


def main():
    rankings = pd.read_csv('socials_rankings.csv')

    num_students, num_sessions = rankings.shape[0], rankings.shape[1] - 1
    sessions = rankings.columns[1:]
                      
    MAX_STUDENTS_PER_SESSION = 20
    NUMBER_OF_ROTATIONS = 3

    # Initialize the optimization model
    problem = pl.LpProblem("FlexDayAssignment", pl.LpMinimize)

    # Define assignment matrix
    # X[i, j, k] = 1 if student i is assigned to session j in rotation k, 0 otherwise
    X = pl.LpVariable.dicts("X", ((i, j, k) for i in range(num_students) for j in range(num_sessions) for k in range(NUMBER_OF_ROTATIONS)), cat=pl.LpBinary)
    
    # Specify contsraints

    # Each student attends exactly one session per rotation
    for i in range(num_students):
        # all sessions available in all but last rotation
        for k in range(NUMBER_OF_ROTATIONS - 1): 
            problem += pl.lpSum(X[i, j, k] for j in range(num_sessions)) == 1, f"one_session_per_student_{i}_rotation_{k}"
        # special case for the last rotation, where the last session is no longer available
        problem += pl.lpSum(X[i, j, NUMBER_OF_ROTATIONS - 1] for j in range(num_sessions - 1)) == 1, f"one_session_per_student_{i}_rotation_{NUMBER_OF_ROTATIONS - 1}"

    # Each student may attend the same session no more than once over all conferences
    for i in range(num_students):
        for j in range(num_sessions):
            problem += pl.lpSum(X[i, j, k] for k in range(NUMBER_OF_ROTATIONS)) <= 1, f"session_{j}_max_once_per_student_{i}"
    # Enrollment cap per session for each rotation
    for j in range(num_sessions):
        for k in range(NUMBER_OF_ROTATIONS):
            # special case - no enrolment in final session of final rotation
            if j == num_sessions - 1 and k == NUMBER_OF_ROTATIONS - 1:
                problem += pl.lpSum(X[i, j, k] for i in range(num_students)) <= 0, f"no_enrolment_in_session_{num_sessions - 1}"
            # enrollment caps for all but last rotation
            else:
                problem += pl.lpSum(X[i, j, k] for i in range(num_students)) <= MAX_STUDENTS_PER_SESSION, f"cap_per_session_{j}_rotation_{k}"
    
    # Create list of lists of rankings data (int)
    R = rankings.iloc[:, 1:].to_numpy().tolist()

    # Set the objective
    objective = pl.lpSum((R[i][j] - 1) * X[i, j, k] for i in range(num_students) for j in range(num_sessions) for k in range(NUMBER_OF_ROTATIONS))
    problem += objective
    
    problem.solve()

    session_enrolments = {k: {j: [] for j in range(num_sessions)} for k in range(NUMBER_OF_ROTATIONS)}
    rankings_of_assigned_sessions_by_student = {i: [] for i in range(num_students)}

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

    for k in range(NUMBER_OF_ROTATIONS):
        pd.DataFrame.from_dict(session_enrolments[k]).set_axis(sessions, axis='columns').to_csv(f"enrolments_for_rotation_{k}.csv", index=False)

    # check to see what students ranked their assigned sessions (measure for how well we did)
    for i, vals in rankings_of_assigned_sessions_by_student.items():
        print(rankings.iloc[i,0] + ":", vals)

if __name__ == '__main__':
    main()

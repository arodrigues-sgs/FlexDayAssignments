import gurobipy as gp
import pandas as pd


def main():
    rankings = pd.read_csv('test.csv')

    num_students, num_sessions = rankings.shape[0], rankings.shape[1] - 1
    sessions = rankings.columns[1:]
                      
    MAX_STUDENTS_PER_SESSION = 4
    NUMBER_OF_ROTATIONS = 3

    # Initialize the optimization model
    model = gp.Model("FlexDayAssignment")

    # Define assignment matrix
    X = model.addVars(num_students, num_sessions, NUMBER_OF_ROTATIONS, vtype=gp.GRB.BINARY, name="assignment_matrix")
    
    # Specify contsraints

    # Each student attends exactly one session per rotation
    for i in range(num_students):
        for k in range(NUMBER_OF_ROTATIONS):
            model.addConstr(sum(X[i, j, k] for j in range(num_sessions)) == 1, name=f"one_session_per_student_{i}_rotation_{k}")

    # Each student may attend the same session no more than once over all conferences
    for i in range(num_students):
        for j in range(num_sessions):
            model.addConstr(sum(X[i, j, k] for k in range(NUMBER_OF_ROTATIONS)) <= 1, f"session_{j}_max_once_per_student_{i}")

    # Enrollment cap per session for each rotation
    for j in range(num_sessions):
        for k in range(NUMBER_OF_ROTATIONS):
            model.addConstr(sum(X[i, j, k] for i in range(num_students)) <= MAX_STUDENTS_PER_SESSION, name=f"cap_per_session_{j}_rotation_{k}")

    # Create list of lists of rankings data (int)
    R = rankings.iloc[:, 1:].to_numpy().tolist()

    objective = gp.quicksum((R[i][j] - 1)**2 * X[i, j, k] for i in range(num_students) for j in range(num_sessions) for k in range(NUMBER_OF_ROTATIONS))

    model.setObjective(objective, gp.GRB.MINIMIZE)

    model.optimize()

    session_enrolments = {k: {j: [] for j in range(num_sessions)} for k in range(NUMBER_OF_ROTATIONS)}

    for (i, j, k), x_ijk in X.items():
        if int(x_ijk.X) == 1:
            session_enrolments[k][j].append(rankings.iloc[i,0])

    # append nones so we can make the output dataframe
    max_length = max(len(lst) for lst in session_enrolments.values())
    
    for k in session_enrolments:
        for j in session_enrolments[k]:
            session_enrolments[k][j] += [None] * (max_length - len(session_enrolments[k][j]))

    for k in range(NUMBER_OF_ROTATIONS):
        pd.DataFrame.from_dict(session_enrolments[k]).set_axis(sessions, axis='columns').to_csv(f"enrolments_for_rotation_{k}.csv", index=False)


if __name__ == '__main__':
    main()

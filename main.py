import gurobipy as gp
import pandas as pd


def main():
    rankings = pd.read_csv('test.csv')

    num_students, num_sessions = rankings.shape[0], rankings.shape[1] - 1
                      
    MAX_STUDENTS_PER_SESSION = 4

    # Initialize the optimization model
    model = gp.Model("FlexDayAssignment")

    # Define assignment matrix
    X = model.addVars(num_students, num_sessions, vtype=gp.GRB.BINARY, name="assignment_matrix")
    
    # Specify contsraints

    # Each student attends exactly one session
    for i in range(num_students):
        model.addConstr(X.sum(i, '*') == 1, name=f"one_session_per_student_{i}")

    # Enrollment cap per session
    for j in range(num_sessions):
        model.addConstr(X.sum('*', j) <= MAX_STUDENTS_PER_SESSION, name=f"cap_per_session_{j}")

    R = rankings.iloc[:, 1:].to_numpy().tolist()

    objective = gp.quicksum((R[i][j] - 1)**2 * X[i, j] for i in range(num_students) for j in range(num_sessions))

    model.setObjective(objective, gp.GRB.MINIMIZE)

    model.optimize()

    assignments = [(i, j) for i in range(num_students) for j in range(num_sessions) if X[i, j].X > 0.5]
    assignments.sort()
    print("Assignments:", assignments)

    sessions = rankings.columns[1:]

    session_enrolments = {j: [] for j in range(num_sessions)}

    for (i, j), x_ij in X.items():
        if int(x_ij.X):
            session_enrolments[j].append(rankings.iloc[i,0])
    
    # append nones so we can make the output dataframe
    max_length = max(len(lst) for lst in session_enrolments.values())
    for j in session_enrolments:
        session_enrolments[j] += [None] * (max_length - len(session_enrolments[j]))

    assignments_df = pd.DataFrame(session_enrolments)
    assignments_df.columns = sessions
    assignments_df.to_csv('output.csv', index=False)


if __name__ == '__main__':
    main()

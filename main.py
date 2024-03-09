import gurobipy as gp
import random
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
    student_ids = rankings.iloc[:,0]
    assign_df = pd.DataFrame(-1, index=student_ids, columns=sessions)

    session_lists = {}

    for (i, j), x_ij in X.items():
        assign_df.iloc[i, j] = int(x_ij.X)
        if int(x_ij.X):
            if sessions[j] not in session_lists.keys():
                session_lists[sessions[j]] = [student_ids[i]]
            else:
                session_lists[sessions[j]].append(student_ids[i])

    print(assign_df)

    for (key, val) in session_lists.items():
        print(key + ":", val)


if __name__ == '__main__':
    main()

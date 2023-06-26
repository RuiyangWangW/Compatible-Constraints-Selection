import numpy as np
import cvxpy as cp
import copy


def chinneck_1 (x_class_0, y_class_0, x_class_1, y_class_1):
    #Define threshold
    eps = 1e-3

    w_combined = cp.Variable((2,1))
    deltas_class_0 = cp.Variable((len(x_class_0),1))
    deltas_class_1 = cp.Variable((len(x_class_1),1))
    A_class_0 = cp.Parameter((len(x_class_0), 2), value=np.zeros((len(x_class_0), 2)))
    A_class_0.value[:,0] = x_class_0
    A_class_0.value[:,1] += 1
    Y_class_0 = cp.Parameter((len(y_class_0), 1), value=np.array(y_class_0).reshape(-1,1))
    A_class_1 = cp.Parameter((len(x_class_1), 2), value=np.zeros((len(x_class_1), 2)))
    A_class_1.value[:,0] = x_class_1
    A_class_1.value[:,1] += 1
    Y_class_1 = cp.Parameter((len(y_class_1), 1), value=np.array(y_class_1).reshape(-1,1))

    constraints = [A_class_0@w_combined>=Y_class_0-deltas_class_0, A_class_1@w_combined<=Y_class_1+deltas_class_1,
                   deltas_class_0 >= np.zeros((len(x_class_0), 1)), deltas_class_1 >= np.zeros((len(x_class_1), 1))]
    obj = cp.Minimize(cp.sum(deltas_class_0)+cp.sum(deltas_class_1))
    constrained_problem = cp.Problem(obj,constraints)

    constrained_problem.solve(solver=cp.GUROBI, reoptimize=True)

    ninf = 0
    sinf = 0
    hold_set_0 = []
    hold_set_1 = []
    for i in range(len(x_class_0)+len(x_class_1)):
        if i < len(x_class_0):
            delta_i = deltas_class_0.value[i]
        else:
            delta_i = deltas_class_1.value[i-len(x_class_0)]

        if delta_i > eps:
            ninf += 1
            sinf += delta_i
            if i < len(x_class_0):
                hold_set_0.append(i)
            else:
                hold_set_1.append(i-len(x_class_0))
    
    num_sens_constraints = len(hold_set_0)+len(hold_set_1)
    

    cover_set_0 = []
    cover_set_1 = []
    if num_sens_constraints ==1:
        cover_set_0 = hold_set_0
        cover_set_1 = hold_set_1
    else:
        MINSINF = 1e6
        while MINSINF > eps:
            MINSINF = 1e6
            candidate = None
            for i in range(num_sens_constraints):
                if i < len(hold_set_0):
                    idx = hold_set_0[i]
                    A_temp = copy.deepcopy(A_class_0.value[idx,:])
                    A_class_0.value[idx,:]=np.zeros((2,))
                    y_temp = copy.deepcopy(Y_class_0.value[idx])
                    Y_class_0.value[idx]=0
                    temp_class = 0
                else:
                    idx = hold_set_1[i-len(hold_set_0)]
                    A_temp = copy.deepcopy(A_class_1.value[idx,:])
                    A_class_1.value[idx,:]=np.zeros((2,))
                    y_temp = copy.deepcopy(Y_class_1.value[idx])
                    Y_class_1.value[idx]=0
                    temp_class = 1

                constrained_problem.solve(solver=cp.GUROBI, reoptimize=True)
                sinf = np.sum(deltas_class_0.value)+np.sum(deltas_class_1.value)
                
                if sinf < MINSINF:
                    MINSINF = sinf
                    candidate = idx
                    candidate_class = temp_class
                    candidate_hold_set_0 = []
                    candidate_hold_set_1 = []
                    for j in range(len(x_class_0)+len(x_class_1)):
                        if j < len(x_class_0):
                            delta_j = deltas_class_0.value[j]
                        else:
                            delta_j = deltas_class_1.value[j-len(x_class_0)]
                        if delta_j > eps:
                            if j < len(x_class_0):
                                candidate_hold_set_0.append(j)
                            else:
                                candidate_hold_set_1.append(j-len(x_class_0))
                if MINSINF < eps:
                    optimal_a = w_combined.value[0]
                    optimal_b = w_combined.value[1]
                    break

                if temp_class == 0:
                    A_class_0.value[idx,:] = A_temp
                    Y_class_0.value[idx] = y_temp
                else:
                    A_class_1.value[idx,:] = A_temp
                    Y_class_1.value[idx] = y_temp
            
            if candidate != None:
                if candidate_class == 0:
                    cover_set_0.append(candidate)
                    A_class_0.value[candidate,:] = np.zeros((2,))
                    Y_class_0.value[candidate] = 0
                else:
                    cover_set_1.append(candidate)
                    A_class_1.value[candidate,:] = np.zeros((2,))
                    Y_class_1.value[candidate] = 0
                hold_set_0 = candidate_hold_set_0
                hold_set_1 = candidate_hold_set_1
                num_sens_constraints = len(hold_set_0) + len(hold_set_1)
                
    result = {
        "cover_set_0" : cover_set_0,
        "cover_set_1" : cover_set_1,
        "optimal_a" : optimal_a,
        "optimal_b" : optimal_b
    }

    return result
    


def chinneck_2 (x_class_0, y_class_0, x_class_1, y_class_1):
    #Define threshold
    eps = 1e-3

    w_combined = cp.Variable((2,1))
    deltas_class_0 = cp.Variable((len(x_class_0),1))
    deltas_class_1 = cp.Variable((len(x_class_1),1))
    A_class_0 = cp.Parameter((len(x_class_0), 2), value=np.zeros((len(x_class_0), 2)))
    A_class_0.value[:,0] = x_class_0
    A_class_0.value[:,1] += 1
    Y_class_0 = cp.Parameter((len(y_class_0), 1), value=np.array(y_class_0).reshape(-1,1))
    A_class_1 = cp.Parameter((len(x_class_1), 2), value=np.zeros((len(x_class_1), 2)))
    A_class_1.value[:,0] = x_class_1
    A_class_1.value[:,1] += 1
    Y_class_1 = cp.Parameter((len(y_class_1), 1), value=np.array(y_class_1).reshape(-1,1))

    constraints = [A_class_0@w_combined>=Y_class_0-deltas_class_0, A_class_1@w_combined<=Y_class_1+deltas_class_1,
                   deltas_class_0 >= np.zeros((len(x_class_0), 1)), deltas_class_1 >= np.zeros((len(x_class_1), 1))]
    obj = cp.Minimize(cp.sum(deltas_class_0)+cp.sum(deltas_class_1))
    constrained_problem = cp.Problem(obj,constraints)

    constrained_problem.solve(solver=cp.GUROBI, reoptimize=True)

    ninf = 0
    sinf = 0
    hold_set_0 = []
    hold_set_1 = []
    for i in range(len(x_class_0)+len(x_class_1)):
        if i < len(x_class_0):
            delta_i = deltas_class_0.value[i]
        else:
            delta_i = deltas_class_1.value[i-len(x_class_0)]

        if delta_i > eps:
            ninf += 1
            sinf += delta_i
            if i < len(x_class_0):
                hold_set_0.append(i)
            else:
                hold_set_1.append(i-len(x_class_0))
    
    num_sens_constraints = len(hold_set_0)+len(hold_set_1)
    

    cover_set_0 = []
    cover_set_1 = []
    if num_sens_constraints ==1:
        cover_set_0 = hold_set_0
        cover_set_1 = hold_set_1
    else:
        MINSINF = 1e6
        MINNINF = 1e6
        while MINSINF > eps:
            MINSINF = 1e6
            MINNINF = 1e6
            candidate = None
            for i in range(num_sens_constraints):
                if i < len(hold_set_0):
                    idx = hold_set_0[i]
                    A_temp = copy.deepcopy(A_class_0.value[idx,:])
                    A_class_0.value[idx,:]=np.zeros((2,))
                    y_temp = copy.deepcopy(Y_class_0.value[idx])
                    Y_class_0.value[idx]=0
                    temp_class = 0
                else:
                    idx = hold_set_1[i-len(hold_set_0)]
                    A_temp = copy.deepcopy(A_class_1.value[idx,:])
                    A_class_1.value[idx,:]=np.zeros((2,))
                    y_temp = copy.deepcopy(Y_class_1.value[idx])
                    Y_class_1.value[idx]=0
                    temp_class = 1

                constrained_problem.solve(solver=cp.GUROBI, reoptimize=True)

                ninf = 0

                for j in range(len(x_class_0)+len(x_class_1)):
                    if j < len(x_class_0):
                        delta_j = deltas_class_0.value[j]
                    else:
                        delta_j = deltas_class_1.value[j-len(x_class_0)]
                    if delta_j > eps:
                        ninf += 1
                
                sinf = np.sum(deltas_class_0.value)+np.sum(deltas_class_1.value)

                if ninf == 0:
                    optimal_a = w_combined.value[0]
                    optimal_b = w_combined.value[1]
                    candidate = idx
                    candidate_class = temp_class
                    MINSINF = sinf
                    MINNINF = 0
                    break
                
                if  (MINNINF > ninf) or (ninf==MINNINF and sinf < MINSINF):
                    MINNINF = ninf
                    MINSINF = sinf
                    candidate = idx
                    candidate_class = temp_class
                    candidate_hold_set_0 = []
                    candidate_hold_set_1 = []
                    for j in range(len(x_class_0)+len(x_class_1)):
                        if j < len(x_class_0):
                            delta_j = deltas_class_0.value[j]
                        else:
                            delta_j = deltas_class_1.value[j-len(x_class_0)]
                        if delta_j > eps:
                            if j < len(x_class_0):
                                candidate_hold_set_0.append(j)
                            else:
                                candidate_hold_set_1.append(j-len(x_class_0))
                    
                if temp_class == 0:
                    A_class_0.value[idx,:] = A_temp
                    Y_class_0.value[idx] = y_temp
                else:
                    A_class_1.value[idx,:] = A_temp
                    Y_class_1.value[idx] = y_temp
            
            if candidate != None:
                if candidate_class == 0:
                    cover_set_0.append(candidate)
                    A_class_0.value[candidate,:] = np.zeros((2,))
                    Y_class_0.value[candidate] = 0

                else:
                    cover_set_1.append(candidate)
                    A_class_1.value[candidate,:] = np.zeros((2,))
                    Y_class_1.value[candidate] = 0
                hold_set_0 = candidate_hold_set_0
                hold_set_1 = candidate_hold_set_1
                num_sens_constraints = len(hold_set_0) + len(hold_set_1)

    result = {
        "cover_set_0" : cover_set_0,
        "cover_set_1" : cover_set_1,
        "optimal_a" : optimal_a,
        "optimal_b" : optimal_b
    }

    return result

            


    


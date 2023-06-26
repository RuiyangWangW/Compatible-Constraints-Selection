import numpy as np
import cvxpy as cp


def optimal_classification(x_class_0, y_class_0, x_class_1, y_class_1):
    #Define Large Value:
    M = cp.Parameter((1,1))
    M.value = np.array([1e4]).reshape(1,1)
    w_combined = cp.Variable((2,1))
    z_0 = cp.Variable((len(x_class_0),1), boolean=True)
    z_1 = cp.Variable((len(x_class_1),1), boolean=True)
    A_class_0 = cp.Parameter((len(x_class_0), 2), value=np.zeros((len(x_class_0), 2)))
    A_class_0.value[:,0] = x_class_0
    A_class_0.value[:,1] += 1
    Y_class_0 = cp.Parameter((len(y_class_0), 1), value=np.array(y_class_0).reshape(-1,1))
    A_class_1 = cp.Parameter((len(x_class_1), 2), value=np.zeros((len(x_class_1), 2)))
    A_class_1.value[:,0] = x_class_1
    A_class_1.value[:,1] += 1
    Y_class_1 = cp.Parameter((len(y_class_1), 1), value=np.array(y_class_1).reshape(-1,1))
    constraints = [A_class_0@w_combined>=Y_class_0-cp.multiply(z_0,M), 
                   A_class_1@w_combined<=Y_class_1+cp.multiply(z_1,M)]
    obj = cp.Minimize(cp.sum(z_0)+cp.sum(z_1))
    constrained_problem = cp.Problem(obj,constraints)

    constrained_problem.solve(solver=cp.GUROBI, reoptimize=True)

    cover_set_0 = []
    cover_set_1 = []
    
    for j in range(len(x_class_0)+len(x_class_1)):
        if j < len(x_class_0):
            if z_0[j].value == 1:
                cover_set_0.append(j)
        else:
            if z_1[j-len(x_class_0)].value == 1:
                cover_set_1.append(j)
    
    optimal_a = w_combined.value[0]
    optimal_b = w_combined.value[1]

    result = {
        "cover_set_0" : cover_set_0,
        "cover_set_1" : cover_set_1,
        "optimal_a" : optimal_a,
        "optimal_b" : optimal_b
    }

    return result

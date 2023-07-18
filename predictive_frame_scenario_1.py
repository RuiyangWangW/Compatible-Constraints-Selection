import numpy as np
from robot_models.SingleIntegrator2D import *
from scipy.stats import norm
import cvxpy as cp
import random
import copy


class predictive_frame_lag:

    def __init__(self, x0, dt, tf, U_max, num_constraints_hard, x_r_list, radius_list, alpha_list, reward_list, obstacle_list, disturbance, disturb_std, disturb_max):
        self.x0 = x0
        self.dt = dt
        self.tf = tf
        self.num_steps = int(self.tf/self.dt)
        self.U_max = U_max
        self.num_constraints_hard = num_constraints_hard
        self.num_constraints_soft = 1
        self.num_constraints_clf = 1
        self.alpha_list = alpha_list
        self.obstacle_list = obstacle_list
        self.reward_list = reward_list
        self.beta_value = 0.6
        self.robot = SingleIntegrator2D(self.x0, self.dt, ax=None, id = 0, color='r', palpha=1.0, \
                                        num_constraints_hard = self.num_constraints_soft+self.num_constraints_hard,
                                        num_constraints_soft = self.num_constraints_clf, plot=False)
        self.disturbance = disturbance
        self.disturb_std = disturb_std
        self.disturb_max = disturb_max
        self.x_r_list = x_r_list
        self.radius_list = radius_list
        self.x_r_id = 0
        self.alpha_clf = 0.8
        self.y_max = 6.0
        self.delta_t_limit = float(self.tf)/len(x_r_list)

    def forward(self):
        
        # Define constrained Optimization Problem
        u1 = cp.Variable((2,1))
        u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
        alpha_soft = cp.Variable((self.num_constraints_soft))
        alpha_0 = cp.Parameter((self.num_constraints_soft))
        h = cp.Parameter((self.num_constraints_soft))
        A1_hard = cp.Parameter((self.num_constraints_hard,2),value=np.zeros((self.num_constraints_hard,2)))
        b1_hard = cp.Parameter((self.num_constraints_hard,1),value=np.zeros((self.num_constraints_hard,1)))
        A1_soft = cp.Parameter((self.num_constraints_soft,2),value=np.zeros((self.num_constraints_soft,2)))
        b1_soft = cp.Parameter((self.num_constraints_soft,1),value=np.zeros((self.num_constraints_soft,1)))
        A1_clf = cp.Parameter((self.num_constraints_clf,2),value=np.zeros((self.num_constraints_clf,2)))
        b1_clf = cp.Parameter((self.num_constraints_clf,1),value=np.zeros((self.num_constraints_clf,1)))
        slack_constraints_clf = cp.Variable((self.num_constraints_clf,1))
        const1 = [A1_hard @ u1 <= b1_hard, A1_soft @ u1 <= b1_soft + cp.multiply(alpha_soft, h), \
                  A1_clf @ u1 <= b1_clf + slack_constraints_clf, cp.norm2(u1) <= self.U_max,
                  alpha_soft >= np.zeros((self.num_constraints_soft)),
                  slack_constraints_clf >= np.zeros((self.num_constraints_clf,1))]
        objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref ) + 100*cp.sum_squares(slack_constraints_clf) 
                                 + 10*cp.sum_squares(alpha_soft-alpha_0))
        constrained_controller = cp.Problem( objective1, const1 ) 
        
        robot = self.robot
        # Define Disturbance 
        u_d = cp.Parameter((2,1), value = np.zeros((2,1)))

        # Define Reward
        r = 0
        reward = 0
        # Define Delta t
        delta_t = 0
        t = 0
        lamda_sum = 0
        lamda_sum_list = []
        flag = "success"
        x_list = []
        y_list = []
        t_list = []
        for i in range(self.num_steps):

            if self.disturbance and robot.X[1]>3.5 and robot.X[0] > -2*self.disturb_std and robot.X[0] < 2*self.disturb_std:
                y_disturb = norm.pdf(robot.X[0], loc=0, scale=self.disturb_std)[0] * self.disturb_max
                x_disturb = 0.0
            else:
                x_disturb = 0.0
                y_disturb = 0.0
     
            u_d.value = np.array([x_disturb, y_disturb]).reshape(2,1)
            x_r = self.x_r_list[self.x_r_id].reshape(2,1)
            alpha_0.value = np.array([self.alpha_list[self.x_r_id]])
            radius = self.radius_list[self.x_r_id]
            v, dv_dx = robot.lyapunov(x_r) 
            robot.A1_soft[0,:] = dv_dx@robot.g()
            robot.b1_soft[0] = -dv_dx@(robot.f()) - self.alpha_clf*v - dv_dx@robot.g()@u_d.value
        
            h1, dh1_dx = robot.static_safe_set(x_r,radius)    
            robot.A1_hard[0,:] = -dh1_dx@robot.g()
            robot.b1_hard[0] = dh1_dx@robot.f() + dh1_dx@robot.g()@u_d.value
            h.value = np.array([h1])

            h2 = (self.y_max - robot.X[1])[0]
            robot.A1_hard[1,:] = np.array([0,1]).reshape(1,2)@robot.g()
            robot.b1_hard[1] = -np.array([0,1]).reshape(1,2)@robot.g()@u_d.value + self.beta_value*h2 - np.array([0,1]).reshape(1,2)@robot.f()

            for j in range(0,len(self.obstacle_list)):
                obs_x_r = self.obstacle_list[j,:].reshape(2,1)
                h_obs, dh_obs_dx = robot.static_safe_set(obs_x_r,0.2) 
                h_obs = -h_obs
                dh_obs_dx = -dh_obs_dx
                robot.A1_hard[j+2,:] = -dh_obs_dx@robot.g()
                robot.b1_hard[j+2] = dh_obs_dx@robot.f() + self.beta_value*h_obs + dh_obs_dx@robot.g()@u_d.value

            A1_clf.value = robot.A1_soft
            b1_clf.value = robot.b1_soft
            A1_soft.value = robot.A1_hard[0,:].reshape(-1,2)
            b1_soft.value = robot.b1_hard[0,:].reshape(-1,1)
            A1_hard.value = robot.A1_hard[1:,:].reshape(-1,2)
            b1_hard.value = robot.b1_hard[1:,:].reshape(-1,1)
            u1_ref.value = robot.nominal_input(x_r)
            
            try:
                constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
                lamda_sum += const1[1].dual_value[0][0]
                u_next = u1.value + u_d.value
                robot.step(u_next)
            except:
                flag = "fail"

            if constrained_controller.status != "optimal" and constrained_controller.status != "optimal_inaccurate":
                flag = "fail"

            delta_t += self.dt
            x_list.append(robot.X[0])
            y_list.append(robot.X[1])
            t += self.dt
            t_list.append(t)

            if delta_t>self.delta_t_limit:
                flag = "fail"

            if flag == "fail":
                lamda_sum_i = copy.deepcopy(lamda_sum)
                lamda_sum_list.append(lamda_sum_i)
                reward = 0
                break

            if (h1 >= 0):
                lamda_sum_i = copy.deepcopy(lamda_sum)
                lamda_sum = 0
                lamda_sum_list.append(lamda_sum_i)
                if self.x_r_id == len(self.x_r_list)-1:
                    break
                reward += self.reward_list[self.x_r_id]
                self.x_r_id += 1
                delta_t = 0
            else:
                continue
        if flag == "success":
            r = 0
        else:
            r = sum(lamda_sum_list)

        return x_list, y_list, t_list, flag, r, reward

def rand_list_init(num_states, idx):
    l = np.ones(shape=(num_states,))
    for i in range(num_states):
        if i == idx:
            l[i] = 0
        else:
            l[i] = random.randint(0,1)
    l = l.tolist()
    return l

def fitness_score_lag(comb, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                  obstacle_list, dt, disturbance, disturb_std, disturb_max,\
                  num_constraints_hard, fitness_score_table):
    
    if fitness_score_table.get(tuple(comb)) != None:
        flag, score, reward = fitness_score_table.get(tuple(comb))
        return flag, score, reward, fitness_score_table

    num_states = len(x_r_list)    
    reward_weight = 0.01
    x_r_list_comb = []
    radii_comb = []
    alpha_list_comb = []
    reward_list_comb = []
    for i in range(num_states):
        if comb[i] == 1:
            x_r_list_comb.append(x_r_list[i])
            radii_comb.append(radius_list[i])
            alpha_list_comb.append(alpha_list[i])
            reward_list_comb.append(reward_list[i])

    if (len(x_r_list_comb)>0):
        pred_frame = predictive_frame_lag(x0,dt,time_horizon,U_max,num_constraints_hard=num_constraints_hard, \
                                    x_r_list=x_r_list_comb, radius_list=radii_comb, alpha_list=alpha_list_comb, \
                                    reward_list = reward_list_comb, obstacle_list=obstacle_list,\
                                    disturbance=disturbance, disturb_std=disturb_std, disturb_max=disturb_max)
        _,_,_, flag, score, reward = pred_frame.forward()
    else:
        reward = 0
    
    score += (reward_max-reward)*reward_weight
    fitness_score_table.update({tuple(comb): [flag, score, reward]})

    return flag, score, reward, fitness_score_table

def mutate_process(comb, mutation_rate):
    mutated_comb = []
    num_states = len(comb)
    for i in range(num_states):
        mutate = np.random.choice([True, False], p=[mutation_rate,1-mutation_rate])
        if mutate:
            mutated_comb.append(random.randint(0,1))
        else:
            mutated_comb.append(comb[i])
    return mutated_comb

def genetic_comb_lag(x0, x_r_list, time_horizon, reward_max, radius_list, alpha_list, reward_list, U_max, obstacle_list, dt, \
                disturbance, disturb_std, disturb_max, num_constraints_hard):

    num_states = len(x_r_list)-1
    num_steps = 2*num_states
    comb_1 = rand_list_init(num_states,2)
    comb_2 = rand_list_init(num_states,3)
    comb_3 = rand_list_init(num_states,4)
    comb_4 = rand_list_init(num_states,6)

    fitness_score_table = {}
    
    comb_1_appended = copy.deepcopy(comb_1)
    comb_1_appended.append(1)
    flag_1, score_1, reward_1,fitness_score_table = fitness_score_lag(comb_1_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
    comb_2_appended = copy.deepcopy(comb_2)
    comb_2_appended.append(1)
    flag_2, score_2, reward_2,fitness_score_table = fitness_score_lag(comb_2_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
    comb_3_appended = copy.deepcopy(comb_3)
    comb_3_appended.append(1)
    flag_3, score_3, reward_3,fitness_score_table = fitness_score_lag(comb_3_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
    comb_4_appended = copy.deepcopy(comb_4)
    comb_4_appended.append(1)    
    flag_4, score_4, reward_4,fitness_score_table = fitness_score_lag(comb_4_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
    
    flag_all = np.array([flag_1, flag_2, flag_3, flag_4])

    fit_all = np.array([score_1, score_2, score_3, score_4]).reshape(4,)

    comb_all = np.array([comb_1,comb_2,comb_3,comb_4])
    
    reward_all = np.array([reward_1, reward_2, reward_3, reward_4]).reshape(4,)
    
    fit_min = np.min(fit_all)
    flag_temp = flag_all[np.argmin(fit_all)]
    comb_min = comb_all[np.argmin(fit_all)].tolist()
    reward_best = reward_all[np.argmin(fit_all)]
    mutation_rate = 0.3
    epsilon = 1e-5
    iteration = 0
    flag_best = "fail"
    while flag_best == "fail":
        iteration += 1
        if flag_temp == "success":
            flag_best = flag_temp
        for i in range(num_steps):
            p = ((1/(fit_all+epsilon)) / np.sum(1/(fit_all+epsilon))).reshape(-1,)
            new_states = np.zeros(comb_all.shape)
            for j in range(0,4):
                new_states[j,:] = comb_all[np.random.choice(a=np.array([0,1,2,3]),p=p)]
            first_split = random.randint(0, num_states)
            comb_1 = np.hstack((new_states[0][0:first_split],new_states[1][first_split:]))
            comb_1 = mutate_process(comb_1,mutation_rate)
            comb_2 = np.hstack((new_states[1][0:first_split],new_states[0][first_split:]))
            comb_2 = mutate_process(comb_2,mutation_rate)

            second_split = random.randint(0, num_states)
            comb_3 = np.hstack((new_states[2][0:second_split],new_states[3][second_split:]))
            comb_3 = mutate_process(comb_3,mutation_rate)
            comb_4 = np.hstack((new_states[3][0:second_split],new_states[2][second_split:]))
            comb_4 = mutate_process(comb_4,mutation_rate)

            comb_1_appended = copy.deepcopy(comb_1)
            comb_1_appended.append(1)
            flag_1, score_1, reward_1,fitness_score_table = fitness_score_lag(comb_1_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                            obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
            comb_2_appended = copy.deepcopy(comb_2)
            comb_2_appended.append(1)
            flag_2, score_2, reward_2,fitness_score_table = fitness_score_lag(comb_2_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                            obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
            comb_3_appended = copy.deepcopy(comb_3)
            comb_3_appended.append(1)
            flag_3, score_3, reward_3,fitness_score_table = fitness_score_lag(comb_3_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                            obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
            comb_4_appended = copy.deepcopy(comb_4)
            comb_4_appended.append(1)    
            flag_4, score_4, reward_4,fitness_score_table = fitness_score_lag(comb_4_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                            obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)

            flag_all = np.array([flag_1, flag_2, flag_3, flag_4])
            fit_all = np.array([score_1, score_2, score_3, score_4]).reshape(4,)
            comb_all = np.array([comb_1,comb_2,comb_3,comb_4])
            reward_all = np.array([reward_1, reward_2, reward_3, reward_4]).reshape(4,)
            if (fit_all.min()<fit_min):
                fit_min = fit_all.min()
                flag_best = flag_all[np.argmin(fit_all)]
                comb_min = comb_all[np.argmin(fit_all)].tolist()
                reward_best = reward_all[np.argmin(fit_all)]

    comb_min.append(1)
    return iteration, comb_min, reward_best

def deterministic_lag(x0, x_r_list, time_horizon, reward_max, radius_list, alpha_list, reward_list,U_max, obstacle_list, dt, \
                disturbance, disturb_std, disturb_max, num_constraints_hard):
    
    init_comb = np.ones(shape=(len(x_r_list)))
    fitness_score_table = {}
    best_flag, min_r, best_reward, fitness_score_table = fitness_score_lag(init_comb, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
    best_comb = init_comb
    dropped_constraints = {}
    while best_flag == "fail":
        min_r = 1.0e6
        for i in range(len(x_r_list)):
            if i!=len(x_r_list)-1 and dropped_constraints.get(i)==None:
                temp_comb = copy.deepcopy(best_comb)
                temp_comb[i] = 0
                flag, r, reward, fitness_score_table = fitness_score_lag(temp_comb, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
                if r <= min_r:
                    min_r = r
                    candidate_idx = i
                    best_reward = reward
                    best_flag = flag
        best_comb[candidate_idx] = 0
        dropped_constraints.update({candidate_idx: True})
        if len(dropped_constraints) == len(x_r_list):
            break
    iteration = 1
    return iteration, best_comb, best_reward
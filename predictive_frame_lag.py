import numpy as np
from robot_models.SingleIntegrator2D import *
from robot_models.DoubleIntegrator2D import *
from scipy.stats import norm
import cvxpy as cp
import random
import math
import copy
import matplotlib.pyplot as plt
from scenario_disturb import *


class predictive_frame_lag:

    def __init__(self, scenario_num, robot_type, x0, dt, tf, U_max, alpha_values, beta_values, num_constraints_hard, x_r_list, radius_list, reward_list, obstacle_list, disturbance, disturb_std, disturb_max):
        self.scenario_num = scenario_num
        self.x0 = x0
        self.dt = dt
        self.tf = tf
        self.num_steps = int(self.tf/self.dt)
        self.U_max = U_max
        self.num_constraints_hard = num_constraints_hard
        self.num_constraints_soft = 1
        self.num_constraints_clf = 1
        self.obstacle_list = obstacle_list
        self.reward_list = reward_list
        if robot_type == 'SingleIntegrator2D':
            self.robot = SingleIntegrator2D(self.x0, self.dt, ax=None, id = 0, color='r', palpha=1.0, \
                                            num_constraints_hard = self.num_constraints_hard,
                                            num_constraints_soft = self.num_constraints_soft, plot=False)
        else:
            self.robot = DoubleIntegrator2D(self.x0, self.dt, ax=None,
                                            num_constraints_hard = self.num_constraints_hard,
                                            num_constraints_soft = self.num_constraints_soft, plot=False)
        self.disturbance = disturbance
        self.disturb_std = disturb_std
        self.disturb_max = disturb_max
        self.f_max_1 = 1/(disturb_std*math.sqrt(2*math.pi))
        self.f_max_2 = self.f_max_1/0.5
        self.x_r_list = x_r_list
        self.radius_list = radius_list
        self.x_r_id = 0
        self.beta_1 = beta_values[0]
        self.beta_2 = beta_values[1]
        self.alpha_1 = alpha_values[0]
        self.alpha_2 = alpha_values[1]
        self.y_max = 6.0
        self.delta_t_limit = self.tf

    def forward(self):
        
        robot = self.robot

        if robot.type == 'SingleIntegrator2D':
            # Define constrained Optimization Problem
            u1 = cp.Variable((2,1))
            u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
            alpha_soft = cp.Variable((self.num_constraints_soft))
            alpha_0 = cp.Parameter((self.num_constraints_soft))
            alpha_0.value = np.array([self.alpha_1])
            v = cp.Parameter((self.num_constraints_soft))
            A1_hard = cp.Parameter((self.num_constraints_hard,2),value=np.zeros((self.num_constraints_hard,2)))
            b1_hard = cp.Parameter((self.num_constraints_hard,1),value=np.zeros((self.num_constraints_hard,1)))
            A1_soft = cp.Parameter((self.num_constraints_soft,2),value=np.zeros((self.num_constraints_soft,2)))
            b1_soft = cp.Parameter((self.num_constraints_soft,1),value=np.zeros((self.num_constraints_soft,1)))
            const1 = [A1_hard @ u1 <= b1_hard, A1_soft @ u1 <= b1_soft + cp.multiply(alpha_soft, v), \
                      cp.norm2(u1) <= self.U_max]
            objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref ) + 10*cp.sum_squares(alpha_soft-alpha_0))
            constrained_controller = cp.Problem( objective1, const1 ) 
        else:
            # Define constrained Optimization Problem
            u1 = cp.Variable((2,1))
            u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
            A1_hard = cp.Parameter((self.num_constraints_hard,2),value=np.zeros((self.num_constraints_hard,2)))
            b1_hard = cp.Parameter((self.num_constraints_hard,1),value=np.zeros((self.num_constraints_hard,1)))
            A1_soft = cp.Parameter((self.num_constraints_soft,2),value=np.zeros((self.num_constraints_soft,2)))
            b1_soft = cp.Parameter((self.num_constraints_soft,1),value=np.zeros((self.num_constraints_soft,1)))
            slack_soft = cp.Variable((self.num_constraints_soft,1))
            const1 = [A1_hard @ u1 <= b1_hard, 
                      A1_soft @ u1 <= b1_soft + slack_soft,
                      slack_soft >= np.zeros((self.num_constraints_soft)), 
                      cp.norm2(u1) <= self.U_max
                      ]
            objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref ))
            constrained_controller = cp.Problem( objective1, const1 ) 
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
     
            u_d.value = disturb_value(robot, self.disturbance, self.disturb_std, self.disturb_max, self.f_max_1, self.f_max_2, scenario_num=self.scenario_num)
            x_r = self.x_r_list[self.x_r_id].reshape(2,1)
            
            radius = self.radius_list[self.x_r_id]


            if robot.type == 'SingleIntegrator2D':
                V, dv_dx = robot.lyapunov(x_r) 
                h1, _ = robot.static_safe_set(x_r, radius)
                h1 = -h1
                robot.A1_soft[0,:] = -dv_dx@robot.g()
                robot.b1_soft[0] = dv_dx@robot.f() + dv_dx@robot.g()@u_d.value
                v.value = np.array([V])

                for j in range(0,len(self.obstacle_list)):
                    obs_x_r = self.obstacle_list[j,:].reshape(2,1)
                    h_obs, dh_obs_dx = robot.static_safe_set(obs_x_r,0.2) 
                    robot.A1_hard[j,:] = -dh_obs_dx@robot.g()
                    robot.b1_hard[j] = dh_obs_dx@robot.f() + self.beta_1*h_obs + dh_obs_dx@robot.g()@u_d.value

            else:
                phi_0, dphi_0_dx, dx12_dt = robot.lyapunov(x_r)
                robot.A1_soft[0,:] = -dphi_0_dx.T@robot.J()
                robot.b1_soft[0] = dphi_0_dx.T@robot.J()@u_d.value - 2*robot.X[2:4].T@dx12_dt - \
                                   (self.alpha_1+self.alpha_2)*dphi_0_dx.T@dx12_dt - (self.alpha_1*self.alpha_2)*phi_0
                h1, _, _ = robot.barrier(x_r, radius)
                h1 = -h1

                for j in range(0,len(self.obstacle_list)):
                    obs_x_r = self.obstacle_list[j,:].reshape(2,1)
                    h, dh_dx, dx12_dt = robot.barrier(obs_x_r,0.2)
                    robot.A1_hard[j,:] = -dh_dx.T@robot.J()
                    robot.b1_hard[j] = dh_dx.T@robot.J()@u_d.value + 2*robot.X[2:4].T@dx12_dt + \
                    (self.beta_1+self.beta_2)*dh_dx.T@dx12_dt + (self.beta_1*self.beta_2)*h

            A1_soft.value = robot.A1_soft
            b1_soft.value = robot.b1_soft
            A1_hard.value = robot.A1_hard.reshape(-1,2)
            b1_hard.value = robot.b1_hard.reshape(-1,1)
            u1_ref.value = robot.nominal_input(x_r)

            try:
                constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
                lamda_sum += const1[1].dual_value[0][0]
                if robot.type == 'SingleIntegrator2D':
                    u_next = u1.value + u_d.value
                    robot.step(u_next)
                else:
                    robot.step(u1.value, u_d.value, self.disturbance)
            except Exception as error:
                flag = "fail"
                break

            if constrained_controller.status != "optimal" and constrained_controller.status != "optimal_inaccurate":
                flag = "fail"
                break

            delta_t += self.dt
            x_list.append(robot.X[0])
            y_list.append(robot.X[1])
            t += self.dt
            t_list.append(t)

            if (h1 >= 0):
                lamda_sum_i = copy.deepcopy(lamda_sum)
                lamda_sum = 0
                lamda_sum_list.append(lamda_sum_i)
                if self.x_r_id == len(self.x_r_list)-1:
                    break
                reward += self.reward_list[self.x_r_id]
                self.x_r_id += 1
                self.delta_t_limit -= delta_t
                delta_t = 0
            else:
                continue
        
        if self.x_r_id < len(self.x_r_list)-1:
            flag = "fail"

        if flag == "fail":
            lamda_sum_i = copy.deepcopy(lamda_sum)
            lamda_sum_list.append(lamda_sum_i)
            reward = 0
                
        if flag == "success":
            r = 0
        else:
            r = sum(lamda_sum_list)

        return x_list, y_list, t_list, flag, r, reward

def rand_list_init(num_states):
    l = np.ones(shape=(num_states,))
    for i in range(num_states):
        l[i] = random.randint(0,1)
    l = l.tolist()
    return l

def fitness_score_lag(comb, scenario_num, robot_type, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_values, beta_values, \
                    reward_list, U_max, obstacle_list, dt, disturbance, disturb_std, disturb_max,\
                    num_constraints_hard, fitness_score_table):
    
    if fitness_score_table.get(tuple(comb)) != None:
        traj, score, reward = fitness_score_table.get(tuple(comb))
        return traj, score, reward, fitness_score_table

    num_states = len(x_r_list)    
    reward_weight = 0.01
    x_r_list_comb = []
    radii_comb = []
    reward_list_comb = []
    for i in range(num_states):
        if comb[i] == 1:
            x_r_list_comb.append(x_r_list[i])
            radii_comb.append(radius_list[i])
            reward_list_comb.append(reward_list[i])

    if (len(x_r_list_comb)>0):
        pred_frame = predictive_frame_lag(scenario_num,robot_type,x0,dt, time_horizon, U_max, alpha_values, beta_values,
                                          num_constraints_hard=num_constraints_hard, x_r_list=x_r_list_comb, radius_list=radii_comb, \
                                    reward_list = reward_list_comb, obstacle_list=obstacle_list,\
                                    disturbance=disturbance, disturb_std=disturb_std, disturb_max=disturb_max)
        x_list, y_list, t_list, flag, score, reward = pred_frame.forward()
        if flag == "success":
            traj = {"x": x_list, "y": y_list, "t": t_list}
        else:
            traj = {}
    else:
        reward = 0
        traj = {}
    
    score += (reward_max-reward)*reward_weight
    fitness_score_table.update({tuple(comb): [traj, score, reward]})

    return traj, score, reward, fitness_score_table

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

def genetic_comb_lag(scenario_num, robot_type, x0, x_r_list, time_horizon, reward_max, radius_list, alpha_list, reward_list, U_max, alpha_clf, beta, obstacle_list, dt, \
                disturbance, disturb_std, disturb_max, num_constraints_hard):

    num_comb = 4
    num_states = len(x_r_list)-1
    num_steps = 2*num_states
    comb_all = []
    traj_all = []
    fit_all = []
    reward_all = []
    for i in range(num_comb):
        comb = rand_list_init(num_states)
        comb_new = copy.deepcopy(comb)
        comb_all.append(comb_new)

    fitness_score_table = {}
    
    for i in range(num_comb):
        comb_appended = copy.deepcopy(comb_all[i])
        comb_appended.append(1)
        traj, score, reward, fitness_score_table = fitness_score_lag(comb_appended, scenario_num, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    alpha_clf, beta, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
        traj_all.append(traj)
        fit_all.append(score)
        reward_all.append(reward)
    
    fit_all = np.array(fit_all)
    fit_min = np.min(fit_all)
    traj_temp = traj_all[np.argmin(fit_all)]
    comb_min = comb_all[np.argmin(fit_all)]
    reward_best = reward_all[np.argmin(fit_all)]
    mutation_rate = 0.3
    epsilon = 1e-5
    iteration = 0
    init_run = True
    while init_run == True or traj_temp == {}:
        iteration += 1
        init_run = False
        for i in range(num_steps):
            p = ((1/(fit_all+epsilon)) / np.sum(1/(fit_all+epsilon))).reshape(-1,)
            new_states = np.zeros([num_comb,num_states])
            for j in range(0,num_comb):
                new_states[j,:] = comb_all[np.random.choice(a=np.arange(0,num_comb,1),p=p)]

            new_comb_all = []

            for k in range(0,int(num_comb/2)):
                split = random.randint(0, num_states)
                comb_1 = np.hstack((new_states[k][0:split],new_states[k+1][split:]))
                comb_1 = mutate_process(comb_1,mutation_rate)
                new_comb_1 = copy.deepcopy(comb_1)
                comb_2 = np.hstack((new_states[k+1][0:split],new_states[k][split:]))
                comb_2 = mutate_process(comb_2,mutation_rate)
                new_comb_2 = copy.deepcopy(comb_2)
                new_comb_all.append(new_comb_1)
                new_comb_all.append(new_comb_2)
            
            comb_all = new_comb_all
            traj_all = []
            fit_all = []
            reward_all = []
            
            for i in range(num_comb):
                comb_appended = copy.deepcopy(comb_all[i])
                comb_appended.append(1)
                traj, score, reward, fitness_score_table = fitness_score_lag(comb_appended, scenario_num, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                            alpha_clf, beta, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
                traj_all.append(traj)
                fit_all.append(score)
                reward_all.append(reward)
            
            fit_all = np.array(fit_all)

            if (fit_all.min()<fit_min):
                fit_min = fit_all.min()
                traj_temp = traj_all[np.argmin(fit_all)]
                comb_min = comb_all[np.argmin(fit_all)]
                reward_best = reward_all[np.argmin(fit_all)]

    comb_min.append(1)
    return iteration, comb_min, traj_temp, reward_best

def deterministic_lag(scenario_num, robot_type, x0, x_r_list, time_horizon, reward_max, radius_list, alpha_values, beta_values, reward_list, U_max, obstacle_list, dt, \
                disturbance, disturb_std, disturb_max, num_constraints_hard):
    
    init_comb = np.ones(shape=(len(x_r_list)))
    fitness_score_table = {}
    traj, min_r, best_reward, fitness_score_table = fitness_score_lag(init_comb, scenario_num, robot_type, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_values, beta_values, 
                                                                      reward_list, U_max, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
    best_comb = init_comb
    best_traj = traj
    dropped_constraints = {}
    while best_traj == {}:
        min_r = 1.0e6
        for i in range(len(x_r_list)):
            if i!=len(x_r_list)-1 and dropped_constraints.get(i)==None:
                temp_comb = copy.deepcopy(best_comb)
                temp_comb[i] = 0
                traj, r, reward, fitness_score_table = fitness_score_lag(temp_comb, scenario_num, robot_type, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_values, beta_values, 
                                                                      reward_list, U_max, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table)
                if best_traj == {}:
                    if r < min_r:
                        min_r = r
                        candidate_idx = i
                        best_reward = reward
                        best_traj = traj
                else:
                    if traj != {} and r < min_r:
                        min_r = r
                        candidate_idx = i
                        best_reward = reward
                        best_traj = traj
        best_comb[candidate_idx] = 0
        dropped_constraints.update({candidate_idx: True})
        if len(dropped_constraints) == len(x_r_list)-1:
            break
    iteration = 1
    return iteration, best_comb, best_traj, best_reward, 
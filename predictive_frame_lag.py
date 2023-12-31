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
    """
    The class that initialize the robot structure and QP based Controller Structure
    """

    def __init__(self, scenario_num, robot_type, x0, dt, tf, U_max, V_max, alpha_values, beta_values, num_constraints_hard, x_r_list, wpt_radius, t_list, reward_list, obstacle_list, if_disturb, disturb_std, disturb_max):
        self.scenario_num = scenario_num
        self.x0 = x0
        self.t = 0.0
        self.dt = dt
        self.tf = tf
        self.num_steps = int(self.tf/self.dt)
        self.U_max = U_max
        self.num_constraints_hard = num_constraints_hard
        self.num_constraints_soft = 1
        self.obstacle_list = obstacle_list
        self.reward_list = reward_list
        self.reward_max = sum(reward_list)
        self.beta_1 = beta_values[0]
        self.beta_2 = beta_values[1]
        self.alpha_1 = alpha_values[0]
        self.alpha_2 = alpha_values[1]

        if robot_type == 'SingleIntegrator2D':
            self.robot = SingleIntegrator2D(self.x0, self.dt, ax=None, id = 0, color='r', palpha=1.0, \
                                            num_constraints_hard = self.num_constraints_hard,
                                            num_constraints_soft = self.num_constraints_soft, plot=False)
        else:
            self.robot = DoubleIntegrator2D(self.x0, self.dt, ax=None, V_max = V_max,
                                            num_constraints_hard = self.num_constraints_hard,
                                            num_constraints_soft = self.num_constraints_soft, plot=False)
            
        if robot_type == 'SingleIntegrator2D':
            # Define constrained Optimization Problem
            self.u1 = cp.Variable((2,1))
            #self.u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
            self.alpha_soft = cp.Variable((1,1))
            self.alpha_1_param = cp.Parameter((1,1))
            self.alpha_1_param.value = np.array([self.alpha_1]).reshape(-1,1)
            self.V = cp.Parameter((self.num_constraints_soft,1))
            self.A1_hard = cp.Parameter((self.num_constraints_hard,2),value=np.zeros((self.num_constraints_hard,2)))
            self.b1_hard = cp.Parameter((self.num_constraints_hard,1),value=np.zeros((self.num_constraints_hard,1)))
            self.A1_soft = cp.Parameter((self.num_constraints_soft,2),value=np.zeros((self.num_constraints_soft,2)))
            self.b1_soft = cp.Parameter((self.num_constraints_soft,1),value=np.zeros((self.num_constraints_soft,1)))
            self.const1 = [self.A1_hard @ self.u1 <= self.b1_hard, self.A1_soft @ self.u1 <= self.b1_soft + cp.multiply(self.alpha_soft, self.V), \
                      self.alpha_soft >= np.zeros((self.num_constraints_soft)), cp.norm2(self.u1) <= self.U_max]
            self.objective1 = cp.Minimize(cp.sum_squares(self.u1) + cp.sum_squares(self.alpha_soft-self.alpha_1_param))
            self.constrained_controller = cp.Problem(self.objective1, self.const1) 
            
            self.circle_A = cp.Parameter((self.num_constraints_hard+self.num_constraints_soft, 2))
            self.circle_A_norm = cp.Parameter((self.num_constraints_hard+self.num_constraints_soft,1))
            self.circle_b = cp.Parameter((self.num_constraints_hard+self.num_constraints_soft, 1))
            self.circle_c = cp.Variable((2,1))
            self.circle_r = cp.Variable((1,1))
            self.circle_obj = cp.Maximize(self.circle_r)
            self.circle_const = [self.circle_A@self.circle_c+cp.multiply(self.circle_r,self.circle_A_norm) <= self.circle_b]
            self.circle_opt = cp.Problem(self.circle_obj, self.circle_const)
        else:
            # Define constrained Optimization Problem
            self.u1 = cp.Variable((2,1))
            #self.u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
            self.alpha_1_param = cp.Parameter((1,1))
            self.alpha_1_param.value = np.array([self.alpha_1]).reshape(-1,1)
            self.alpha_2_param = cp.Parameter((1,1))
            self.alpha_2_param.value = np.array([self.alpha_2]).reshape(-1,1)
            self.alpha_soft = cp.Variable((1,1))
            self.beta_1_param = cp.Parameter((1,1))
            self.beta_1_param.value = np.array([self.beta_1]).reshape(-1,1)
            self.beta_2_param = cp.Parameter((1,1))
            self.beta_2_param.value = np.array([self.beta_2]).reshape(-1,1)
            self.beta_hard = cp.Variable((1,1))
            self.H = cp.Parameter((self.num_constraints_hard,1))
            self.H.value = np.zeros((self.num_constraints_hard,1))
            self.dH_dx_T = cp.Parameter((num_constraints_hard,2))
            self.dH_dx_T.value = np.zeros((self.num_constraints_hard,2))
            self.phi = cp.Parameter((self.num_constraints_soft,1))
            self.dphi_dx_T = cp.Parameter((1,2))
            self.dx12_dt = cp.Parameter((2,1))
            self.A1_hard = cp.Parameter((self.num_constraints_hard,2),value=np.zeros((self.num_constraints_hard,2)))
            self.b1_hard = cp.Parameter((self.num_constraints_hard,1),value=np.zeros((self.num_constraints_hard,1)))
            self.A1_soft = cp.Parameter((self.num_constraints_soft,2),value=np.zeros((self.num_constraints_soft,2)))
            self.b1_soft = cp.Parameter((self.num_constraints_soft,1),value=np.zeros((self.num_constraints_soft,1)))
            self.slack_soft = cp.Variable((self.num_constraints_soft,1))
            self.const1 = [self.A1_hard @ self.u1 <= self.b1_hard + cp.multiply(self.beta_1_param+self.beta_hard, self.dH_dx_T@self.dx12_dt) +
                        cp.multiply(cp.multiply(self.beta_1_param,self.beta_hard), self.H), 
                        self.A1_soft @ self.u1 <= self.b1_soft + cp.multiply(self.alpha_1_param+self.alpha_soft, self.dphi_dx_T@self.dx12_dt) +
                        cp.multiply(cp.multiply(self.alpha_1_param,self.alpha_soft), self.phi),
                        self.alpha_soft >= np.zeros((1,1)), self.beta_hard >= np.zeros((1,1)),
                        cp.norm2(self.u1[0]) <= self.U_max, cp.norm2(self.u1[1]) <= self.U_max]
            self.objective1 = cp.Minimize(cp.sum_squares(self.u1)+cp.sum_squares(self.alpha_soft-self.alpha_2_param)+cp.sum_squares(self.beta_hard-self.beta_2_param))
            self.constrained_controller = cp.Problem(self.objective1, self.const1)
        # Define if_disturb 
        self.u_d = cp.Parameter((2,1), value = np.zeros((2,1)))
        self.if_disturb = if_disturb
        self.disturb_std = disturb_std
        self.disturb_max = disturb_max
        self.f_max_1 = 1/(self.disturb_std*math.sqrt(2*math.pi))
        self.f_max_2 = self.f_max_1*2.0
        self.x_r_list = x_r_list
        self.wpt_radius = wpt_radius
        self.t_list = t_list
        self.x_r_id = 0
        self.y_max = 6.0
        self.eps = 1e-5

    def forward(self, x0, curr_t, sub_list):
        """
        Given the robot position, current time, and a list of waypoint indices

        The forward algorithm will generate a trajectory if the QP optimization problem is feasible
        in all time steps starting from curr_t to curr_t + self.tf. 

        If the QP optimization problem become infesaible at any time step, the algorithm will return a list
        of Lagrange variables.
        """
        #Initialization
        self.x0 = x0
        self.robot.X = x0
        self.t = curr_t
        arrived_waypoint = 0
        self.x_r_id = sub_list[arrived_waypoint]
        curr_wpt = self.x_r_id

        # Define Reward
        reward = 0

        # Define lambda_sum
        lamda_sum = 0
        #step_used = 0
        lamda_sum_list = []

        flag = "success"
        if self.robot.type == 'SingleIntegrator2D':
            x_list = np.zeros((2,self.num_steps))
        else:
            x_list = np.zeros((4,self.num_steps))
        t_list = np.zeros((self.num_steps))

        for i in range(self.num_steps):
     
            self.u_d.value = disturb_value(self.robot, self.if_disturb, self.disturb_std, self.disturb_max, self.f_max_1, self.f_max_2, scenario_num=self.scenario_num)
            x_r = self.x_r_list[self.x_r_id].reshape(2,1)
            
            radius = self.wpt_radius

            if self.robot.type == 'SingleIntegrator2D':
                v, dv_dx = self.robot.lyapunov(x_r) 
                h1, _ = self.robot.static_safe_set(x_r, radius)
                h1 = -h1
                self.robot.A1_soft[0,:] = -dv_dx@self.robot.g()
                self.robot.b1_soft[0] = dv_dx@self.robot.f() + dv_dx@self.robot.g()@self.u_d.value
                self.V.value = np.array([v]).reshape(-1,1)

                for j in range(0,len(self.obstacle_list)):
                    obs_x_r = self.obstacle_list[j,:].reshape(2,1)
                    h_obs, dh_obs_dx = self.robot.static_safe_set(obs_x_r,0.4)
                    self.robot.A1_hard[j,:] = -dh_obs_dx@self.robot.g()
                    self.robot.b1_hard[j] = dh_obs_dx@self.robot.f() + self.beta_1*h_obs + dh_obs_dx@self.robot.g()@self.u_d.value

            else:
                phi_0, dphi_0_dx, dx12_dt = self.robot.lyapunov(x_r)
                self.phi.value = phi_0
                self.dphi_dx_T.value = dphi_0_dx.T
                self.dx12_dt.value = dx12_dt
                self.robot.A1_soft[0,:] = -dphi_0_dx.T@self.robot.J()
                self.robot.b1_soft[0] = dphi_0_dx.T@self.u_d.value - 2*dx12_dt.T@dx12_dt
                h1, _, _ = self.robot.barrier(x_r, radius)
                h1 = -h1
                
                for j in range(0,len(self.obstacle_list)):
                    obs_x_r = self.obstacle_list[j,:].reshape(2,1)
                    h, dh_dx, dx12_dt = self.robot.barrier(obs_x_r,0.4)
                    self.H.value[j] = h
                    self.dH_dx_T.value[j,:] = dh_dx.T
                    self.robot.A1_hard[j,:] = -dh_dx.T@self.robot.J()
                    self.robot.b1_hard[j] = dh_dx.T@self.u_d.value + 2*dx12_dt.T@dx12_dt
                
                
            self.A1_soft.value = self.robot.A1_soft
            self.b1_soft.value = self.robot.b1_soft
            self.A1_hard.value = self.robot.A1_hard.reshape(-1,2)
            self.b1_hard.value = self.robot.b1_hard.reshape(-1,1)

            try:
                self.constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
                #if self.const1[1].dual_value[0][0] >= self.eps:
                #    lamda_sum += 1
                lamda_sum += abs(self.const1[1].dual_value[0][0])
                #prev_lamda = self.const1[1].dual_value[0][0]
                #prev_alpha_soft = self.alpha_soft.value
                
                #self.robot.b1_soft += self.alpha_soft.value * v
                #circle_A = np.append(self.robot.A1_soft,self.robot.A1_hard,axis=0)
                #circle_b = np.append(self.robot.b1_soft,self.robot.b1_hard,axis=0)
                #self.circle_A_norm.value = np.linalg.norm(circle_A, axis=1).reshape(-1,1)
                #self.circle_A.value = circle_A
                #self.circle_b.value = circle_b 
                #self.circle_opt.solve(solver=cp.GUROBI, reoptimize=True)
                #lamda_sum += -self.circle_r.value
            except Exception as error:
                #print(error)
                flag = "fail"
                break
             

            if self.constrained_controller.status != "optimal" and self.constrained_controller.status != "optimal_inaccurate":
                flag = "fail"
                break

            if self.robot.type == 'SingleIntegrator2D':
                u_next = self.u1.value + self.u_d.value
                #prev_u = u_next
                self.robot.step(u_next)
                x_list[:,i] = self.robot.X.reshape(2,)
            else:
                self.robot.step(self.u1.value, self.u_d.value, self.if_disturb)
                x_list[:,i] = self.robot.X.reshape(4,)

            curr_t += self.dt
           
            t_list[i] = curr_t

            if (curr_t > self.t_list[self.x_r_id] and h1<0):
                flag = "fail"
                break
    
            if (h1 >= 0):
                #lamda_sum /= (i-step_used)
                #step_used = i
                lamda_sum_i = copy.deepcopy(lamda_sum)
                lamda_sum = 0
                lamda_sum_list.append(lamda_sum_i)
                reward += self.reward_list[self.x_r_id]
                arrived_waypoint += 1
                if arrived_waypoint == len(sub_list):
                    curr_wpt = len(self.x_r_list)
                    break
                self.x_r_id = sub_list[arrived_waypoint]
                curr_wpt = self.x_r_id
            
            else:
                continue

        if flag == "fail":
            #pseudo_v, dv_dx = self.robot.lyapunov(x_r)
            #self.robot.A1_soft[0,:] = -dv_dx@self.robot.g()
            #self.robot.b1_soft[0] = dv_dx@self.robot.f() + dv_dx@self.robot.g()@self.u_d.value
            #pseudo_h = self.robot.A1_soft@prev_u - prev_alpha_soft*pseudo_v
            #lamda_sum = pseudo_h[0][0]
            lamda_sum_i = copy.deepcopy(lamda_sum)
            lamda_sum_list.append(lamda_sum_i)
            reward = 0
            i -= 1
                

        x_list = x_list[:,0:i]
        t_list = t_list[0:i]

        return x_list, t_list, flag, lamda_sum_list, reward, curr_wpt

def rand_list_init(num_states):
    l = np.ones(shape=(num_states,))
    for i in range(num_states):
        l[i] = random.randint(0,1)
    l = l.tolist()
    return l

def fitness_score_lag(comb, x0, curr_t, pred_frame):

    num_states = len(comb)    
    #reward_weight = 0.01
    #reward_weight = 10.0
    reward_weight = 1.0
    sub_list = []
    for i in range(num_states):
        if comb[i] == 1:
           sub_list.append(i)

    if (len(sub_list)>0):
        x_list, t_list, flag, lambda_sum_list, reward, curr_wpt = pred_frame.forward(x0, curr_t, sub_list)
        if flag == "success":
            traj = {"x": x_list, "t": t_list}
        else:
            traj = {}
    else:
        reward = 0
        traj = {}
    
    score = (pred_frame.reward_max-reward)*reward_weight + sum(lambda_sum_list)

    return traj, score, reward, curr_wpt

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

def genetic_comb_lag(x0, curr_time, pred_frame):

    num_comb = 6
    num_states = len(pred_frame.x_r_list)-1
    num_steps = 2*num_states
    comb_all = []
    traj_all = []
    fit_all = []
    reward_all = []
    curr_wpt_all = []

    for i in range(num_comb):
        comb = rand_list_init(num_states)
        comb_new = copy.deepcopy(comb)
        comb_all.append(comb_new)
    
    for i in range(num_comb):
        comb_appended = copy.deepcopy(comb_all[i])
        comb_appended.append(1)
        traj, score, reward, curr_wpt = fitness_score_lag(comb_appended, x0, curr_time, pred_frame)
        traj_all.append(traj)
        fit_all.append(score)
        reward_all.append(reward)
        curr_wpt_all.append(curr_wpt)
    
    fit_all = np.array(fit_all)
    fit_min = np.min(fit_all)
    traj_temp = traj_all[np.argmin(fit_all)]
    comb_min = comb_all[np.argmin(fit_all)]
    reward_best = reward_all[np.argmin(fit_all)]
    curr_wpt_best = curr_wpt_all[np.argmin(fit_all)]
    mutation_rate = 0.3
    epsilon = 1e-5
    init_run = True
    while init_run == True or traj_temp == {}:
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
            curr_wpt_all = []
            
            for i in range(num_comb):
                comb_appended = copy.deepcopy(comb_all[i])
                comb_appended.append(1)
                traj, score, reward, curr_wpt = fitness_score_lag(comb_appended, x0, curr_time, pred_frame)
                traj_all.append(traj)
                fit_all.append(score)
                reward_all.append(reward)
                curr_wpt_all.append(curr_wpt)

            
            fit_all = np.array(fit_all)

            if (fit_all.min()<fit_min):
                fit_min = fit_all.min()
                traj_temp = traj_all[np.argmin(fit_all)]
                comb_min = comb_all[np.argmin(fit_all)]
                reward_best = reward_all[np.argmin(fit_all)]
                curr_wpt_best = curr_wpt_all[np.argmin(fit_all)]

    comb_min.append(1)
    return curr_wpt_best, comb_min, traj_temp, reward_best

def deterministic_lag(x0, curr_time, pred_frame):
    
    num_states = len(pred_frame.x_r_list)
    init_comb = np.ones(shape=(num_states))
    traj, r, reward, curr_wpt = fitness_score_lag(init_comb, x0, curr_time, pred_frame)
    comb_best = init_comb
    traj_best = traj
    r_min = r
    reward_best = reward
    curr_wpt_best = curr_wpt

    dropped_constraints = {}
    while traj_best == {}:
        r_min = 1.0e6
        for i in range(num_states):
            if i!=num_states-1 and dropped_constraints.get(i)==None:
                temp_comb = copy.deepcopy(comb_best)
                temp_comb[i] = 0
                traj, r, reward, curr_wpt= fitness_score_lag(temp_comb, x0, curr_time, pred_frame)
                if traj_best == {}:
                    if r < r_min:
                        r_min = r
                        candidate_idx = i
                        reward_best = reward
                        traj_best = traj
                        curr_wpt_best = curr_wpt
                else:
                    if traj != {} and r < r_min:
                        r_min = r
                        candidate_idx = i
                        reward_best = reward
                        traj_best = traj
                        curr_wpt_best = curr_wpt
        comb_best[candidate_idx] = 0
        dropped_constraints.update({candidate_idx: True})
        if len(dropped_constraints) == num_states-1:
            break
    return curr_wpt_best, comb_best, traj_best, reward_best


def greedy_lag(x0, curr_time, pred_frame):

    num_states = len(pred_frame.x_r_list)
    comb_best = np.ones(shape=(num_states))
    sub_list = []
    for i in range(num_states):
        if comb_best[i] == 1:
            sub_list.append(i)
    
    dropped_constraints = {}

    x_list, t_list, flag, lambda_sum_list, reward, curr_wpt = pred_frame.forward(x0, curr_time, sub_list)
    
    while flag == "fail":
        if len(dropped_constraints) == num_states-1:
            break
        candidate_idx = np.argmax(np.array([lambda_sum_list]))
        if sub_list[candidate_idx] == num_states-1:
            lambda_sum_list[candidate_idx] = -1e5
            candidate_idx = np.argmax(np.array([lambda_sum_list]))
        if (dropped_constraints.get(i)==None):
            comb_best[sub_list[candidate_idx]] = 0
            dropped_constraints.update({sub_list[candidate_idx]: True})
        sub_list = []
        for i in range(num_states):
            if comb_best[i] == 1:
                sub_list.append(i)
        x_list, t_list, flag, lambda_sum_list, reward, curr_wpt = pred_frame.forward(x0, curr_time, sub_list)
    
    curr_wpt_best = curr_wpt
    traj_best = {"x": x_list, "t": t_list}
    reward_best = reward
    return curr_wpt_best, comb_best, traj_best, reward_best
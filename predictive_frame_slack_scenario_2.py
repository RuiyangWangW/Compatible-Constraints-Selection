import numpy as np
from robot_models.SingleIntegrator2D import *
from scipy.stats import norm
import cvxpy as cp
import random
import copy

class predictive_frame_slack:

    def __init__(self, x0, dt, tf, U_max, num_constraints_hard, reward_list, x_r_list, radius_list, alpha_list, obstacle_list, disturbance, disturb_std, disturb_max):
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
        self.robot = SingleIntegrator2D(self.x0, self.dt, ax=None, id = 0, color='r', palpha=1.0, \
                                        num_constraints_hard = self.num_constraints_soft+self.num_constraints_hard,
                                        num_constraints_soft = self.num_constraints_clf, plot=False)
        self.disturbance = disturbance
        self.disturb_std = disturb_std
        self.disturb_max = disturb_max
        self.x_r_list = x_r_list
        self.radius_list = radius_list
        self.reward_list = reward_list
        self.x_r_id = 0
        self.alpha_clf = 0.8
        self.beta_value = 0.6
        self.y_max = 6.0
        self.delta_t_limit = float(self.tf)/len(x_r_list)
        self.eps = 1e-5

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


        # Define relaxed Optimization Problem
        u_relaxed = cp.Variable((2,1))
        alpha_soft = cp.Variable((self.num_constraints_soft))
        slack_constraints_hard = cp.Variable((self.num_constraints_hard,1))
        slack_constraints_soft = cp.Variable((self.num_constraints_soft,1))
        slack_constraints_clf = cp.Variable((self.num_constraints_clf,1))
        const_relaxed = [A1_hard @ u_relaxed <= b1_hard + slack_constraints_hard, A1_soft @ u_relaxed <= b1_soft + cp.multiply(alpha_soft, h) + slack_constraints_soft, \
                  A1_clf @ u_relaxed <= b1_clf + slack_constraints_clf, cp.norm2(u_relaxed) <= self.U_max,
                  alpha_soft >= np.zeros((self.num_constraints_soft)),
                  slack_constraints_hard >= np.zeros((self.num_constraints_hard,1)),
                  slack_constraints_soft >= np.zeros((self.num_constraints_soft,1)),
                  slack_constraints_clf >= np.zeros((self.num_constraints_clf,1)),]
        objective_relaxed = cp.Minimize(cp.sum_squares(u_relaxed - u1_ref ) + 100*cp.sum_squares(slack_constraints_clf) 
                                 + 1000*cp.sum_squares(slack_constraints_soft) + 10*cp.sum_squares(alpha_soft-alpha_0)
                                 + 1000*cp.sum_squares(slack_constraints_hard))
        relaxed_controller = cp.Problem(objective_relaxed, const_relaxed) 
        
        robot = self.robot
        # Define Disturbance 
        u_d = cp.Parameter((2,1), value = np.zeros((2,1)))

        # Define Reward
        r = 0
        reward = 0
        # Define Delta t
        delta_t = 0
        slack_total_sum = 0
        flag = "success"
        for i in range(self.num_steps):

            if self.disturbance and robot.X[1]>3.5 and robot.X[0] > -2*self.disturb_std and robot.X[0] < 2*self.disturb_std:
                y_disturb = norm.pdf(self.robot.X[0], loc=0, scale=self.disturb_std)[0] * self.disturb_max
                x_disturb = 0.0
            elif self.disturbance and robot.X[0]>-0.5 and robot.X[0] < 1.8\
                and robot.X[1] > -2*(self.disturb_std*0.5) and robot.X[1] < 2*(self.disturb_std*0.5):
                x_disturb = norm.pdf(self.robot.X[1], loc=0, scale=self.disturb_std*0.5)[0] * self.disturb_max
                y_disturb = 0.0
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
                u_next = u1.value + u_d.value
                robot.step(u_next)
            except:
                relaxed_controller.solve(solver=cp.GUROBI, reoptimize=True)
                flag = "fail"

            if constrained_controller.status != "optimal":
                relaxed_controller.solve(solver=cp.GUROBI, reoptimize=True)
                flag = "fail"

            delta_t += self.dt

            if (h1 < 0) and (delta_t>self.delta_t_limit):
                relaxed_controller.solve(solver=cp.GUROBI, reoptimize=True)
                flag = "fail"

            if flag == "fail":
                if slack_constraints_soft.value[0][0] > self.eps:
                    slack_total_sum = slack_constraints_soft.value[0][0]

                for hard_slack_idx in range(self.num_constraints_hard):
                    if slack_constraints_hard.value[hard_slack_idx] > self.eps:
                        slack_total_sum += slack_constraints_hard.value[hard_slack_idx][0]

                reward = 0
                r = slack_total_sum
                break

            if ((h1 >= 0) and (delta_t<self.delta_t_limit)):
                if self.x_r_id == len(self.x_r_list)-1:
                    break
                if (h1 >= 0) and (delta_t<self.delta_t_limit):
                    reward += self.reward_list[self.x_r_id]
                self.x_r_id += 1
                delta_t = 0
            else:
                continue

        return r, reward

def fitness_score_slack(comb, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                  obstacle_list, dt, disturbance, disturb_std, disturb_max,\
                  num_constraints_hard, fitness_score_table, mode):
    
    if fitness_score_table.get(tuple(comb)) != None:
        if mode == 'deterministic':
            score, active_set, reward = fitness_score_table.get(tuple(comb))
            return score, active_set, reward, fitness_score_table
        else:
            score, raw_score, reward = fitness_score_table.get(tuple(comb))
            return score, raw_score, reward, fitness_score_table

    num_states = len(x_r_list)    
    reward_weight = 0.01
    x_r_list_comb = []
    radii_comb = []
    alpha_list_comb = []
    reward_list_comb = []
    active_set = []
    for i in range(num_states):
        if comb[i] == 1:
            active_set.append(i)
            x_r_list_comb.append(x_r_list[i])
            radii_comb.append(radius_list[i])
            alpha_list_comb.append(alpha_list[i])
            reward_list_comb.append(reward_list[i])

    if (len(x_r_list_comb)>0):
        pred_frame = predictive_frame_slack(x0,dt,time_horizon,U_max,num_constraints_hard=num_constraints_hard,reward_list=reward_list_comb, \
                                    x_r_list=x_r_list_comb, radius_list=radii_comb, alpha_list=alpha_list_comb, obstacle_list=obstacle_list,\
                                    disturbance=disturbance, disturb_std=disturb_std, disturb_max=disturb_max)
        score, reward = pred_frame.forward()
    else:
        reward = 0
    if mode != 'deterministic':
        raw_score = copy.deepcopy(score)
        #score += (reward_max-reward)*reward_weight
        score = (reward_max-reward)
        fitness_score_table.update({tuple(comb): [score, raw_score, reward]})
        return score, raw_score, reward, fitness_score_table
    
    fitness_score_table.update({tuple(comb): [score, active_set, reward]})
    return score, active_set, reward, fitness_score_table

def deterministic_chinneck_1(x0, x_r_list, time_horizon, reward_max, radius_list, alpha_list, reward_list, U_max, obstacle_list, dt, \
                disturbance, disturb_std, disturb_max, num_constraints_hard):
    
    init_comb = np.ones(shape=(len(x_r_list)))
    eps = 1.0e-5
    fitness_score_table = {}
    min_r, active_set, reward, fitness_score_table = fitness_score_slack(init_comb, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, \
                                    U_max, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table, mode='deterministic')
    best_comb = init_comb
    while abs(min_r) > eps:
        min_r = 1.0e5
        for i in range(len(active_set)):
            if active_set[i]!=len(x_r_list)-1:
                temp_comb = copy.deepcopy(best_comb)
                temp_comb[active_set[i]] = 0
                r, temp_active_set, reward_temp,fitness_score_table = fitness_score_slack(temp_comb, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, \
                                    U_max, obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table, mode='deterministic')
                if r <= min_r:
                    min_r = r
                    candidate_idx = active_set[i]
                    candidate_active_set = temp_active_set
                    reward = reward_temp
        if active_set[0]==len(x_r_list)-1:
            reward = 0
            break
        best_comb[candidate_idx] = 0
        active_set = candidate_active_set        
    iteration = 1
    return iteration, best_comb, reward

def rand_list_init(num_states, idx):
    l = np.ones(shape=(num_states,))
    for i in range(num_states):
        if i == idx:
            l[i] = 0
        else:
            l[i] = random.randint(0,1)
    l = l.tolist()
    return l


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

def genetic_comb_slack(x0, x_r_list, time_horizon, reward_max, radius_list, alpha_list, reward_list, U_max, obstacle_list, dt, \
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
    score_1, raw_score_1, reward_1, fitness_score_table = fitness_score_slack(comb_1_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table, 'evolve')
    comb_2_appended = copy.deepcopy(comb_2)
    comb_2_appended.append(1)
    score_2, raw_score_2, reward_2, fitness_score_table = fitness_score_slack(comb_2_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table, 'evolve')
    comb_3_appended = copy.deepcopy(comb_3)
    comb_3_appended.append(1)
    score_3, raw_score_3, reward_3, fitness_score_table = fitness_score_slack(comb_3_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table, 'evolve')
    comb_4_appended = copy.deepcopy(comb_4)
    comb_4_appended.append(1)    
    
    score_4, raw_score_4, reward_4, fitness_score_table = fitness_score_slack(comb_4_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table, 'evolve')
    
    fit_all = np.array([score_1, score_2, score_3, score_4]).reshape(4,)

    comb_all = np.array([comb_1,comb_2,comb_3,comb_4])
    fit_min = np.min(fit_all)
    comb_min = comb_all[np.argmin(fit_all)].tolist()
    raw_score_all = np.array([raw_score_1, raw_score_2, raw_score_3, raw_score_4]).reshape(4,)
    reward_all = np.array([reward_1, reward_2, reward_3, reward_4]).reshape(4,)
    raw_score_min = raw_score_all[np.argmin(fit_all)]
    reward = reward_all[np.argmin(fit_all)]

    mutation_rate = 0.3
    epsilon = 1e-5
    no_viable_sol = True
    iteration = 1

    while no_viable_sol:
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

            comb_all = np.array([comb_1,comb_2,comb_3,comb_4])

            comb_1_appended = copy.deepcopy(comb_1)
            comb_1_appended.append(1)        
            score_1, raw_score_1, reward_1, fitness_score_table = fitness_score_slack(comb_1_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table, 'evolve')
            comb_2_appended = copy.deepcopy(comb_2)
            comb_2_appended.append(1)
            score_2, raw_score_2, reward_2, fitness_score_table = fitness_score_slack(comb_2_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table, 'evolve')
            comb_3_appended = copy.deepcopy(comb_3)
            comb_3_appended.append(1)
            score_3, raw_score_3, reward_3, fitness_score_table = fitness_score_slack(comb_3_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table, 'evolve')
            comb_4_appended = copy.deepcopy(comb_4)
            comb_4_appended.append(1)
            score_4, raw_score_4, reward_4, fitness_score_table = fitness_score_slack(comb_4_appended, x0, time_horizon, reward_max, x_r_list, radius_list, alpha_list, reward_list, U_max, \
                                    obstacle_list, dt, disturbance, disturb_std, disturb_max, num_constraints_hard, fitness_score_table, 'evolve')

            fit_all = np.array([score_1, score_2, score_3, score_4]).reshape(4,)
            raw_score_all = np.array([raw_score_1, raw_score_2, raw_score_3, raw_score_4]).reshape(4,)
            reward_all = np.array([reward_1, reward_2, reward_3, reward_4]).reshape(4,)

            if (fit_all.min()<fit_min):
                fit_min = fit_all.min()
                comb_min = comb_all[np.argmin(fit_all)].tolist()
                raw_score_min = raw_score_all[np.argmin(fit_all)]
                reward = reward_all[np.argmin(fit_all)]

        print("raw score min: ", raw_score_min)
        if raw_score_min < epsilon:
            no_viable_sol = False
        else:
            iteration += 1

    comb_min.append(1)
    return iteration, comb_min, reward
import numpy as np
import math
import time
import cvxpy as cp
import copy
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
from queue import PriorityQueue
from robot_models.SingleIntegrator2D import *
from Safe_Set_Series import *
from matplotlib.animation import FFMpegWriter
from Trajectory_Model import *
from predictive_frame_scenario_1 import *
from predictive_frame_slack_scenario_1 import *

t_start = time.perf_counter()
plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.1
t = 0
tf = 60
num_steps = int(tf/dt)

# Define Parameters for CLF and CBF
U_max = 1.0
d_max = 0.6
beta_value = 0.6
alpha_clf = 0.8
num_constraints_soft1 = 1
num_constraints_clf = 1
# Plot                  
plt.ion()
x_min = -6
x_max = 6
y_min = -2
y_max = 6
fig = plt.figure()
ax = plt.axes(xlim=(x_min,x_max),ylim=(y_min,y_max+2)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
# Define Series of Safe Sets
num_points = 7
centroids = PointsInCircum(r=5,n=(num_points*2))[1:num_points+1]

rect = patches.Rectangle((-5, y_max), 10, 0.5, linewidth=1, edgecolor='none', facecolor='k')
# Add the patch to the Axes
ax.add_patch(rect)
radii = np.zeros((centroids.shape[0],))+d_max
alpha_list = np.zeros((centroids.shape[0],))+1.0
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids,radii=radii,alpha_list=alpha_list)

obstacle_list = np.array([])

num_constraints_hard1 = obstacle_list.shape[0] + 1

radii = np.zeros((centroids.shape[0],))+d_max
alpha_list = np.zeros((centroids.shape[0],))+0.4
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids,radii=radii,alpha_list=alpha_list)

reward_list = np.array([1,1,1,1,1,1,0])
reward_max = np.sum(reward_list)

for i in range(0,centroids.shape[0]):
    if i != centroids.shape[0]-1:
        circle = patches.Circle(centroids[i,:], radius=radii[i], color='blue', zorder=0)
    else:
        circle = patches.Circle(centroids[i,:], radius=radii[i], color='red', zorder=0)
    ax.add_patch(circle)
ax.axis('equal')

#Define Disturbance
disturbance = True
disturb_std = 1.5
disturb_max = 6.0 * U_max

x_disturb_1 = np.arange(start=-2*disturb_std, stop=2*disturb_std+0.1, step=0.1)
y_disturb_1 = norm.pdf(x_disturb_1, loc=0, scale=disturb_std) * disturb_max + 3.5
ax.plot(x_disturb_1, y_disturb_1, linewidth=0.5, color='b')

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
movie_name = 'series_of_safesets_small_wind_cbf_online.mp4'


best_idx = 0
best_reward = 0

x0 = np.array([5.0,0.0])

total_reward = 0
total_iter = 0
for i in range(1):
    iteration, best_comb, reward = deterministic_chinneck_1(x0=x0, x_r_list=centroids, time_horizon=tf, reward_max=reward_max,radius_list=radii, \
                                                              alpha_list=alpha_list, reward_list=reward_list, U_max = U_max, dt=dt, disturbance=disturbance, \
                                                            disturb_std=disturb_std, disturb_max=disturb_max, obstacle_list=obstacle_list, \
                                                            num_constraints_hard=num_constraints_hard1)
    x_r_list_comb = []
    radius_list_comb = []
    alpha_list_comb = []
    for k in range(len(best_comb)):
        if best_comb[k] == 1:
            x_r_list_comb.append(centroids[k,:])
            radius_list_comb.append(radii[k])
            alpha_list_comb.append(alpha_list[k])

    total_reward += reward
    num_iter = iteration
    total_iter += num_iter
    print('Reward: ', reward )
    print('Time (s): ', time.perf_counter()-t_start)

print('Average Reward: ', total_reward/10.0)
print('Average Iter: ', total_iter/10.0)


# Define constrained Optimization Problem
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
alpha_soft = cp.Variable((num_constraints_soft1))
alpha_0 = cp.Parameter((num_constraints_soft1))
h = cp.Parameter((num_constraints_soft1))
A1_hard = cp.Parameter((num_constraints_hard1,2),value=np.zeros((num_constraints_hard1,2)))
b1_hard = cp.Parameter((num_constraints_hard1,1),value=np.zeros((num_constraints_hard1,1)))
A1_soft = cp.Parameter((num_constraints_soft1,2),value=np.zeros((num_constraints_soft1,2)))
b1_soft = cp.Parameter((num_constraints_soft1,1),value=np.zeros((num_constraints_soft1,1)))
A1_clf = cp.Parameter((num_constraints_clf,2),value=np.zeros((num_constraints_clf,2)))
b1_clf = cp.Parameter((num_constraints_clf,1),value=np.zeros((num_constraints_clf,1)))
slack_constraints_clf = cp.Variable((num_constraints_clf,1))
const1 = [A1_hard @ u1 <= b1_hard, A1_soft @ u1 <= b1_soft + cp.multiply(alpha_soft, h), \
          A1_clf @ u1 <= b1_clf + slack_constraints_clf, cp.norm2(u1) <= U_max,
          alpha_soft >= np.zeros((num_constraints_soft1)),
          slack_constraints_clf >= np.zeros((num_constraints_clf,1))]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref ) + 100*cp.sum_squares(slack_constraints_clf) 
                         + 10*cp.sum_squares(alpha_soft-alpha_0))
constrained_controller = cp.Problem( objective1, const1 ) 
# Define Disturbance 
u_d = cp.Parameter((2,1), value = np.zeros((2,1)))

# Define Robot
robot = SingleIntegrator2D(x0, dt, ax=ax, id = 0, color='r', palpha=1.0, num_constraints_hard = num_constraints_hard1 + num_constraints_soft1, \
                            num_constraints_soft = num_constraints_clf, plot=True)
# Define Disturbance 
u_d = cp.Parameter((2,1), value = np.zeros((2,1)))
delta_t_limit = float(tf)/len(x_r_list_comb)
x_list = []
y_list = []
t_list = []
t = 0
id_updated = True
#with writer.saving(fig, movie_name, 100):
x_r_id = 0
delta_t = 0
flag = "success"
for i in range(num_steps):

    
    # Define Disturbance 
    u_d = cp.Parameter((2,1), value = np.zeros((2,1)))
    # Define Delta t
    delta_t = 0
    t = 0
    flag = "success"

    if disturbance and robot.X[1]>3.5 and robot.X[0] > -2*disturb_std and robot.X[0] < 2*disturb_std:
        y_disturb = norm.pdf(robot.X[0], loc=0, scale=disturb_std)[0] * disturb_max
        x_disturb = 0.0
    else:
        x_disturb = 0.0
        y_disturb = 0.0
     
    u_d.value = np.array([x_disturb, y_disturb]).reshape(2,1)
    x_r = x_r_list_comb[x_r_id].reshape(2,1)
    alpha_0.value = np.array([alpha_list_comb[x_r_id]])
    radius = radius_list_comb[x_r_id]
    v, dv_dx = robot.lyapunov(x_r) 
    robot.A1_soft[0,:] = dv_dx@robot.g()
    robot.b1_soft[0] = -dv_dx@(robot.f()) - alpha_clf*v - dv_dx@robot.g()@u_d.value
        
    h1, dh1_dx = robot.static_safe_set(x_r,radius)    
    robot.A1_hard[0,:] = -dh1_dx@robot.g()
    robot.b1_hard[0] = dh1_dx@robot.f() + dh1_dx@robot.g()@u_d.value
    h.value = np.array([h1])

    h2 = (y_max - robot.X[1])[0]
    robot.A1_hard[1,:] = np.array([0,1]).reshape(1,2)@robot.g()
    robot.b1_hard[1] = -np.array([0,1]).reshape(1,2)@robot.g()@u_d.value + beta_value*h2 - np.array([0,1]).reshape(1,2)@robot.f()

    for j in range(0,len(obstacle_list)):
        obs_x_r = obstacle_list[j,:].reshape(2,1)
        h_obs, dh_obs_dx = robot.static_safe_set(obs_x_r,0.2) 
        h_obs = -h_obs
        dh_obs_dx = -dh_obs_dx
        robot.A1_hard[j+2,:] = -dh_obs_dx@robot.g()
        robot.b1_hard[j+2] = dh_obs_dx@robot.f() + beta_value*h_obs + dh_obs_dx@robot.g()@u_d.value

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
        flag = "fail"

    if constrained_controller.status != "optimal" and constrained_controller.status != "optimal_inaccurate":
        flag = "fail"

    delta_t += dt
    x_list.append(robot.X[0])
    y_list.append(robot.X[1])
    t += dt
    t_list.append(t)

    if delta_t>delta_t_limit:
        flag = "fail"

    if flag == "fail":
        print("fail")
        break

    if (h1 >= 0):
        if x_r_id == len(x_r_list_comb)-1:
            break
        else:
            circle = patches.Circle(x_r, radius=radius, color='green', zorder=2)
            ax.add_patch(circle)
        x_r_id += 1
        delta_t = 0
    else:
        continue


plt.ioff()
im = ax.scatter(x_list,y_list,cmap='copper',c=t_list, zorder=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('time(s) colorbar')
plt.show()



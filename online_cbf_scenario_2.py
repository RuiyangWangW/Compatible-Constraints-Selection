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
from predictive_frame_lag import *
from predictive_frame_slack import *

t_start = time.perf_counter()
plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.1
t = 0
#tf = 120 # for Single Integrator 
t_horizon = 60
final_wpt_time = 60
# Define Parameters for CLF and CBF
U_max = 1.0
#U_max = 2.0
d_max = 0.6
V_max = 1.0
alpha_values = [2.0, 0.0] 
beta_values = [1.0, 0]
#alpha_values = [10, 10]
#beta_values = [10, 10]
robot_type = 'SingleIntegrator2D'
scenario_num = 2
num_constraints_soft1 = 1
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
x_r_list = scenario_waypoints(scenario_num,robot_type)
rect = patches.Rectangle((-4.9, y_max-0.2), 9.8, 0.4, linewidth=1, edgecolor='none', facecolor='k')
obstacle_list_x_1 = np.arange(start=-4.8+0.1,stop=4.8+0.1, step=0.2)
obstacle_list_y_1 = np.zeros(shape=obstacle_list_x_1.shape)+6.0
obstacle_list_1 = np.vstack((obstacle_list_x_1,obstacle_list_y_1)).T
# Add the patch to the Axes
ax.add_patch(rect)
rect = patches.Rectangle((-2.9, 0.8), 2.2, 0.4, linewidth=1, edgecolor='none', facecolor='k')
obstacle_list_x_2 = np.arange(start=-2.8+0.1,stop=-0.8+0.1, step=0.2)
obstacle_list_y_2 = np.zeros(shape=obstacle_list_x_2.shape)+1.0
obstacle_list_2 = np.vstack((obstacle_list_x_2,obstacle_list_y_2)).T
# Add the patch to the Axes
ax.add_patch(rect)
rect = patches.Rectangle((0.8, -0.9), 0.4, 2.2, linewidth=1, edgecolor='none', facecolor='k')
ax.add_patch(rect)
obstacle_list_y_3 = np.arange(start=-0.8+0.1, stop=1.2+0.1, step=0.2)
obstacle_list_x_3 = np.zeros(shape=obstacle_list_y_3.shape)+1.0
obstacle_list_3 = np.vstack((obstacle_list_x_3,obstacle_list_y_3)).T
obstacle_list = np.vstack((obstacle_list_1,obstacle_list_2))
obstacle_list = np.vstack((obstacle_list,obstacle_list_3))

num_constraints_hard1 = obstacle_list.shape[0]

for i in range(0,obstacle_list.shape[0]):
    circle = patches.Circle(obstacle_list[i,:], radius=0.2, color='black', zorder=0)
    ax.add_patch(circle)
ax.axis('equal')


radii = np.zeros((x_r_list.shape[0],))+d_max
Safe_Set_Series = Safe_Set_Series2D(centroids=x_r_list,radii=radii)

reward_list = np.array([1,1,1,1,1,1,1,1,0])
t_step = int(final_wpt_time/len(reward_list))
t_list = np.arange(t_step, final_wpt_time+t_step, t_step)
reward_max = np.sum(reward_list)

for i in range(0,x_r_list.shape[0]):
    if i != x_r_list.shape[0]-1:
        circle = patches.Circle(x_r_list[i,:], radius=radii[i], color='blue', zorder=0)
    else:
        circle = patches.Circle(x_r_list[i,:], radius=radii[i], color='red', zorder=0)
    ax.add_patch(circle)
ax.axis('equal')

#Define Disturbance
if_disturb = False
disturb_max = 1.25*U_max
disturb_std = 1.5
f_max_1 = 1/(disturb_std*math.sqrt(2*math.pi))
f_max_2 = f_max_1*2.0


if if_disturb:
    if robot_type != 'DoubleIntegrator2D':
        x_disturb_1 = np.arange(start=-2*disturb_std, stop=2*disturb_std+0.1, step=0.1)
        y_disturb_1 = norm.pdf(x_disturb_1, loc=0, scale=disturb_std)/f_max_1 * disturb_max + 4.0
        ax.fill_between(x_disturb_1, y_disturb_1, 4.0, alpha=0.2, color='blue')

        y_disturb_2 = np.arange(start=-2*(disturb_std*0.5), stop=2*(disturb_std*0.5)+0.1, step=0.1)
        x_disturb_2 = norm.pdf(y_disturb_2, loc=0, scale=disturb_std*0.5)/f_max_2 * disturb_max - 0.5
        ax.fill_betweenx(y_disturb_2,x_disturb_2,-0.5, alpha=0.2, color='blue')
    else:
        x_disturb_1 = np.arange(start=-2*disturb_std, stop=2*disturb_std+0.1, step=0.1)
        y_disturb_1 = norm.pdf(x_disturb_1, loc=0, scale=disturb_std)/f_max_1 * disturb_max + 4.0
        ax.fill_between(x_disturb_1, y_disturb_1, 4.0, alpha=0.2, color='blue')

        y_disturb_2 = np.arange(start=-2*(disturb_std*0.5), stop=2*(disturb_std*0.5)+0.1, step=0.1)
        x_disturb_2 = norm.pdf(y_disturb_2, loc=0, scale=disturb_std*0.5)/f_max_2 * disturb_max - 1.5
        ax.fill_betweenx(y_disturb_2,x_disturb_2,-1.5, alpha=0.2, color='blue')

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
movie_name = 'series_of_safesets_small_wind_cbf_online.mp4'


best_idx = 0
best_reward = 0


if robot_type == 'DoubleIntegrator2D': 
    x0 = np.array([5.0,2.0,0.0,0.0]).reshape(4,1)
else:
    x0 = np.array([5.0,2.0]).reshape(2,1)

total_reward = 0
total_iter = 0

pred_frame = predictive_frame_lag(scenario_num=scenario_num, robot_type=robot_type, x0=x0, dt=dt, tf=t_horizon, U_max=U_max,
                                  V_max=V_max, alpha_values=alpha_values, beta_values=beta_values, num_constraints_hard=num_constraints_hard1,
                                  x_r_list=x_r_list,wpt_radius=d_max,t_list=t_list,reward_list=reward_list,obstacle_list=obstacle_list,
                                  if_disturb=if_disturb, disturb_std=disturb_std, disturb_max=disturb_max)



comb_best = np.ones(shape=(x_r_list.shape[0],))
curr_wpt_best = 0
curr_t = 0
x_list = []
y_list = []
t_list = []
while curr_wpt_best < x_r_list.shape[0]:
    adaptive_x_r_list = []
    adaptive_t_list = []
    adaptive_reward_list = []
    for i in range(curr_wpt_best, len(comb_best), 1):
        if comb_best[i] == 1:
            adaptive_x_r_list.append(pred_frame.x_r_list[i])
            adaptive_t_list.append(pred_frame.t_list[i])
            adaptive_reward_list.append(pred_frame.reward_list[i])

    if len(adaptive_x_r_list) == 0:
        break

    pred_frame.x_r_list = adaptive_x_r_list
    pred_frame.t_list = adaptive_t_list
    pred_frame.reward_list = adaptive_reward_list
    
    curr_wpt_best, comb_best, traj_best, reward_best = deterministic_lag(x0=x0, curr_time=curr_t, pred_frame=pred_frame)

    if traj_best["x"].shape[1] == 0:
        break

    if len(x_list) == 0:
        x_list = traj_best["x"][0,:]
        y_list = traj_best["x"][1,:]
        t_list = traj_best["t"]
    else:
        x_list = np.hstack([x_list,traj_best["x"][0,:]])
        y_list = np.hstack([y_list,traj_best["x"][1,:]])
        t_list = np.hstack([t_list,traj_best["t"]])
    if robot_type == 'DoubleIntegrator2D':  
        x0 = traj_best["x"][:,-1].reshape(4,1)
    else:
        x0 = traj_best["x"][:,-1].reshape(2,1)
    curr_t = t_list[-1]

    total_reward += reward_best

    for i in range(0, curr_wpt_best):
        if comb_best[i] == 1:
            centroid = adaptive_x_r_list[i]
            r = radii[i]
            circle = patches.Circle(centroid, radius=r, color='green', zorder=2)
            ax.add_patch(circle)
            

print('Reward: ', total_reward )

t_end = time.perf_counter()

print('Calculation Time: ', t_end - t_start)

plt.ioff()
im = ax.scatter(x_list,y_list,cmap='copper',c=t_list, zorder=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('time(s) colorbar')

plt.show()
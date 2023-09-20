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
tf = 60
num_steps = int(tf/dt)
final_wpt_time = 60

# Define Parameters for CLF and CBF
U_max = 1.0
d_max = 0.6
alpha_0 = 1.2
alpha_clf = 0.4
beta = 1.8
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
centroids[0][0] = 4.5
centroids[0][1] = 1.0

centroids[1][0] = 3.0
centroids[1][1] = 3.0

centroids[2][1] = 5.0
centroids[2][0] = 0

centroids[3][0] = -1.5
centroids[3][1] = 3.0

centroids[4][0] = centroids[2][0]
centroids[4][1] = 0

centroids[5][0] = -4.0
centroids[5][1] = 0.0

centroids[6][0] = -4.0
centroids[6][1] = 3.0

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
obstacle_list = np.vstack((obstacle_list_1,obstacle_list_2))

num_constraints_hard1 = obstacle_list.shape[0] + 1

for i in range(0,obstacle_list.shape[0]):
    circle = patches.Circle(obstacle_list[i,:], radius=0.2, color='black', zorder=0)
    ax.add_patch(circle)
ax.axis('equal')


radii = np.zeros((centroids.shape[0],))+d_max
alpha_list = np.zeros((centroids.shape[0],))+alpha_0
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids,radii=radii,alpha_list=alpha_list)

reward_list = np.array([1,1,1,1,1,1,0])
reward_max = np.sum(reward_list)
t_step = int(final_wpt_time/len(reward_list))
t_list = np.arange(t_step, final_wpt_time+t_step, t_step)

for i in range(0,centroids.shape[0]):
    if i != centroids.shape[0]-1:
        circle = patches.Circle(centroids[i,:], radius=radii[i], color='blue', zorder=0)
    else:
        circle = patches.Circle(centroids[i,:], radius=radii[i], color='red', zorder=0)
    ax.add_patch(circle)
ax.axis('equal')

#Define Disturbance
disturbance = True
disturb_max = 1.5*U_max
disturb_std = 1.5
f_max_1 = 1/(disturb_std*math.sqrt(2*math.pi))
f_max_2 = f_max_1/0.5


x_disturb_1 = np.arange(start=-2*disturb_std, stop=2*disturb_std+0.1, step=0.1)
y_disturb_1 = norm.pdf(x_disturb_1, loc=0, scale=disturb_std)/f_max_1 * disturb_max + 4.0
ax.fill_between(x_disturb_1, y_disturb_1, 4.0, alpha=0.2, color='blue')
y_disturb_2 = np.arange(start=-2*(disturb_std*0.5), stop=2*(disturb_std*0.5)+0.1, step=0.1)
x_disturb_2 = norm.pdf(y_disturb_2, loc=0, scale=disturb_std*0.5)/f_max_2 * disturb_max - 0.5
ax.fill_betweenx(y_disturb_2,x_disturb_2,-0.5, alpha=0.2, color='blue')

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
movie_name = 'series_of_safesets_small_wind_cbf_online.mp4'


best_idx = 0
best_reward = 0

x0 = np.array([5.0,0.0])

total_reward = 0
total_iter = 0
for i in range(1):
    iteration, best_comb, best_traj, reward = deterministic_lag(scenario_num=1,x0=x0, x_r_list=centroids, time_horizon=tf, reward_max=reward_max,t_list=t_list, \
                                                              alpha_list=alpha_list, reward_list=reward_list, U_max = U_max, wpt_radius=d_max,alpha_clf=alpha_clf, beta=beta, dt=dt, disturbance=disturbance, \
                                                            disturb_std=disturb_std, disturb_max=disturb_max, obstacle_list=obstacle_list, \
                                                            num_constraints_hard=num_constraints_hard1)

    total_reward += reward
    num_iter = iteration
    total_iter += num_iter
    print('Reward: ', reward )
    print('Time (s): ', time.perf_counter()-t_start)

print('Average Reward: ', total_reward/10.0)
print('Average Iter: ', total_iter/10.0)

x_list = best_traj["x"]
y_list = best_traj["y"]
t_list = best_traj["t"]

for i in range(len(best_comb)-1):
    if best_comb[i] == 1:
        centroid = centroids[i,:]
        r = radii[i]
        circle = patches.Circle(centroid, radius=r, color='green', zorder=2)
        ax.add_patch(circle)



plt.ioff()
im = ax.scatter(x_list,y_list,cmap='copper',c=t_list, zorder=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('time(s) colorbar')
plt.show()



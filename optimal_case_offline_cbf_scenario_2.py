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
from predictive_frame_scenario_2 import *

t_start = time.perf_counter()
plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.1
t = 0
tf = 50
num_steps = int(tf/dt)

# Define Parameters for CLF and CBF
U_max = 1.0
d_max = 0.6
beta_list = [0.6]
alpha_clf = 0.4
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
num_points = 9
centroids = PointsInCircum(r=5,n=(num_points*2))[1:num_points+1]
centroids[0][0] = 4.5
centroids[0][1] = 1.0

centroids[1][0] = 3.0
centroids[1][1] = 3.0

centroids[2][0] = 2.5
centroids[2][1] = 5.0

centroids[3][1] = 5.0
centroids[3][0] = 0

centroids[4][0] = -2.5
centroids[4][1] = 5.0

centroids[5][0] = -1.5
centroids[5][1] = 3.0

centroids[6][0] = centroids[3][0]
centroids[6][1] = 0

centroids[7][0] = -4.0
centroids[7][1] = 0.0

centroids[8][0] = -4.0
centroids[8][1] = 3.0


rect = patches.Rectangle((-5, y_max), 10, 0.5, linewidth=1, edgecolor='none', facecolor='k')
# Add the patch to the Axes
ax.add_patch(rect)
rect = patches.Rectangle((-3, 1.0), 2.4, 0.4, linewidth=1, edgecolor='none', facecolor='k')
obstacle_list_x_1 = np.arange(start=-3+0.2,stop=-0.6+0.2, step=0.4)
obstacle_list_y_1 = np.zeros(shape=obstacle_list_x_1.shape)+1.2
obstacle_list_1 = np.vstack((obstacle_list_x_1,obstacle_list_y_1)).T
# Add the patch to the Axes
ax.add_patch(rect)
rect = patches.Rectangle((1, -1), 0.4, 2.4, linewidth=1, edgecolor='none', facecolor='k')
ax.add_patch(rect)
obstacle_list_y_2 = np.arange(start=-1+0.2, stop=1.4+0.2, step=0.4)
obstacle_list_x_2 = np.zeros(shape=obstacle_list_y_2.shape)+1.2
obstacle_list_2 = np.vstack((obstacle_list_x_2,obstacle_list_y_2)).T
obstacle_list = np.vstack((obstacle_list_1,obstacle_list_2))

num_constraints_hard1 = obstacle_list.shape[0] + 1

for i in range(0,obstacle_list.shape[0]):
    circle = patches.Circle(obstacle_list[i,:], radius=0.2, color='black', zorder=0)
    ax.add_patch(circle)
ax.axis('equal')


radii = np.zeros((centroids.shape[0],))+d_max
alpha_list = np.zeros((centroids.shape[0],))+0.4
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids,radii=radii,alpha_list=alpha_list)

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
disturb_max = 2.0 * U_max

x_disturb_1 = np.arange(start=-2*disturb_std, stop=2*disturb_std+0.1, step=0.1)
y_disturb_1 = norm.pdf(x_disturb_1, loc=0, scale=disturb_std) * disturb_max + 3.5
ax.fill_between(x_disturb_1, y_disturb_1, 3.5, alpha=0.2, color='blue')

y_disturb_2 = np.arange(start=-2*(disturb_std*0.5), stop=2*(disturb_std*0.5)+0.1, step=0.1)
x_disturb_2 = norm.pdf(y_disturb_2, loc=0, scale=disturb_std*0.5) * disturb_max - 0.5
ax.fill_betweenx(y_disturb_2,x_disturb_2,-0.5, alpha=0.2, color='blue')



metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
movie_name = 'series_of_safesets_small_wind_cbf.mp4'

all_comb = list(itertools.product([0, 1], repeat=num_points-1))


best_idx = 0
best_reward = 0

x0 = np.array([5.0,0.0])

for idx, comb in enumerate(all_comb):
    x_r_list = []
    radius_list = []
    alpha_list_comb = []

    for i in range(num_points-1):
        if comb[i] == 1:
            x_r_list.append(centroids[i,:])
            radius_list.append(radii[i])
            alpha_list_comb.append(alpha_list[i])
    x_r_list.append(centroids[-1,:])
    radius_list.append(radii[-1])
    alpha_list_comb.append(alpha_list[-1])

    if len(x_r_list) > 0:
        pred_frame = predictive_frame_lag(x0,dt,tf,U_max,num_constraints_hard=num_constraints_hard1,beta_list=beta_list, \
                                    x_r_list=x_r_list, radius_list=radius_list, alpha_list=alpha_list_comb, obstacle_list=obstacle_list,\
                                    disturbance=disturbance, disturb_std=disturb_std, disturb_max=disturb_max)
        x_list_comb, y_list_comb, t_list_comb, score, _ = pred_frame.forward(flag='offline')
    else:
        score = 0
    if score >= best_reward:
        if score == best_reward and comb.count(1) < all_comb[best_idx].count(1):
            continue
        x_list = x_list_comb
        y_list = y_list_comb
        t_list = t_list_comb
        best_reward = score
        best_idx = idx
        
best_comb = all_comb[best_idx]
print(best_reward)
print(time.perf_counter()-t_start)
x_r_list = []
radius_list = []
alpha_list_comb = []
for i in range(len(best_comb)):
    if best_comb[i] == 1:
        x_r_list.append(centroids[i,:])
        radius_list.append(radii[i])
        alpha_list_comb.append(alpha_list[i])

for i in range(len(x_r_list)):
    circle = patches.Circle(x_r_list[i], radius=d_max, color='green', zorder=0)
    ax.add_patch(circle)
ax.axis('equal')

plt.ioff()

im = ax.scatter(x_list,y_list,cmap='copper',c=t_list, zorder=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('time(s) colorbar')
plt.show()
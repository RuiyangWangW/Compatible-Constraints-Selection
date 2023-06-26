import numpy as np
import math
import matplotlib as mpl
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
dt = 0.2
t = 0
tf = 60
num_steps = int(tf/dt)

# Define Parameters for CLF and CBF
U_max = 1.0
d_max = 0.6
beta_list = [0.4]
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
ax = plt.axes(xlim=(x_min,x_max),ylim=(y_min,y_max+1)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
# Define Series of Safe Sets
num_points = 9
centroids = PointsInCircum(r=5,n=(num_points*2))[1:num_points+1]
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

num_constraints_hard1 = obstacle_list.shape[0] + 1

radii = np.zeros((centroids.shape[0],))+d_max
alpha_list = np.zeros((centroids.shape[0],))+0.4
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids,radii=radii,alpha_list=alpha_list)
s = centroids.shape[0]

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
disturb_max = 4.0 * U_max

x_disturb_1 = np.arange(start=-2*disturb_std, stop=2*disturb_std+0.1, step=0.1)
y_disturb_1 = norm.pdf(x_disturb_1, loc=0, scale=disturb_std) * disturb_max + 3.5

ax.fill_between(x_disturb_1, y_disturb_1, 3.5, alpha=0.2, color='blue')

y_disturb_2 = np.arange(start=-2*(disturb_std*0.5), stop=2*(disturb_std*0.5)+0.1, step=0.1)
x_disturb_2 = norm.pdf(y_disturb_2, loc=0, scale=disturb_std*0.5) * disturb_max - 0.5

ax.fill_betweenx(y_disturb_2,x_disturb_2,-0.5, alpha=0.2, color='blue')

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
movie_name = 'series_of_safesets_small_wind_cbf_online.mp4'


best_idx = 0
best_reward = 0

x0 = np.array([5.0,0.0])

# Define Unrelaxed Optimization Problem
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
alpha_soft = cp.Variable((num_constraints_soft1))
alpha_0 = cp.Parameter((num_constraints_soft1))
h = cp.Parameter((num_constraints_soft1))
num_constraints_hard1 = num_constraints_hard1
num_constraints_soft1 = num_constraints_soft1
num_constraints_clf = num_constraints_clf
A1_hard = cp.Parameter((num_constraints_hard1,2),value=np.zeros((num_constraints_hard1,2)))
b1_hard = cp.Parameter((num_constraints_hard1,1),value=np.zeros((num_constraints_hard1,1)))
A1_soft = cp.Parameter((num_constraints_soft1,2),value=np.zeros((num_constraints_soft1,2)))
b1_soft = cp.Parameter((num_constraints_soft1,1),value=np.zeros((num_constraints_soft1,1)))
A1_clf = cp.Parameter((num_constraints_clf,2),value=np.zeros((num_constraints_clf,2)))
b1_clf = cp.Parameter((num_constraints_clf,1),value=np.zeros((num_constraints_clf,1)))
slack_constraints1 = cp.Variable((num_constraints_soft1,1))
const1 = [A1_hard @ u1 <= b1_hard, A1_soft @ u1 <= b1_soft + cp.multiply(alpha_soft, h), \
          A1_clf @ u1 <= b1_clf + slack_constraints1, cp.norm2(u1) <= U_max]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref ) + 1000*cp.sum_squares(slack_constraints1) + 1000*cp.sum_squares(alpha_soft-alpha_0))
constrained_controller = cp.Problem( objective1, const1 ) 


# Define Robot
robot = SingleIntegrator2D(x0, dt, ax=ax, id = 0, color='r', palpha=1.0, num_constraints_hard = num_constraints_hard1 + num_constraints_soft1, \
                            num_constraints_soft = num_constraints_clf, plot=True)
thresh = 0.1
# Define Disturbance 
u_d = cp.Parameter((2,1), value = np.zeros((2,1)))
delta_t_limit = float(tf)/len(centroids)
x_list = []
y_list = []
t_list = []
t = 0
id_updated = True
re_plan = True

time_horizon = 10.0
ava_id_list = []
t_replan = 0
with writer.saving(fig, movie_name, 100):
    x_r_id = 0
    delta_t = 0
    for i in range(num_steps):
        if re_plan:
            iteration, best_comb = genetic_comb_lag(x0=robot.X, x_r_list=centroids[x_r_id:,:], time_horizon=time_horizon,radius_list=radii[x_r_id:],\
                                                    alpha_list=alpha_list[x_r_id:], U_max = U_max, dt=dt, disturbance=disturbance, \
                                disturb_std=disturb_std, disturb_max=disturb_max, beta_value=beta_list[0], obstacle_list=obstacle_list, \
                                    num_constraints_hard=num_constraints_hard1)
            ava_id_list = []
            for i in range(len(best_comb)):
                if best_comb[i] == 1:
                    ava_id = x_r_id + i
                    ava_id_list.append(ava_id)
            t_replan = 0
            re_plan = False
            
        if disturbance and robot.X[1]>4.0 and robot.X[0] > -2*disturb_std and robot.X[0] < 2*disturb_std:
            y_disturb = norm.pdf(robot.X[0], loc=0, scale=disturb_std)[0] * disturb_max
            x_disturb = 0.0
        elif disturbance and robot.X[0]>-0.5 and robot.X[0] < 1.8\
            and robot.X[1] > -2*(disturb_std*0.5) and robot.X[1] < 2*(disturb_std*0.5):
            x_disturb = norm.pdf(robot.X[1], loc=0, scale=disturb_std*0.5)[0] * disturb_max * 2.0
            y_disturb = 0.0
        else:
            x_disturb = 0.0
            y_disturb = 0.0

        if id_updated:
            centroid = centroids[x_r_id]
            r = radii[x_r_id]
            circle = patches.Circle(centroid, radius=r, color='green', zorder=1)
            ax.add_patch(circle)
            id_updated = False

        u_d.value = np.array([x_disturb, y_disturb]).reshape(2,1)
        x_r = centroids[x_r_id].reshape(2,1)
        alpha_0.value = np.array([alpha_list[x_r_id]])
        radius = radii[x_r_id]
        v, dv_dx = robot.lyapunov(x_r) 
        robot.A1_soft[0,:] = dv_dx@robot.g()
        robot.b1_soft[0] = -dv_dx@(robot.f()) - alpha_clf*v - dv_dx@robot.g()@u_d.value
    
        h1, dh1_dx = robot.static_safe_set(x_r,radius)    
        robot.A1_hard[0,:] = -dh1_dx@robot.g()
        robot.b1_hard[0] = dh1_dx@robot.f() + dh1_dx@robot.g()@u_d.value
        h.value = np.array([h1])

        h2 = (y_max - robot.X[1])[0]
        robot.A1_hard[1,:] = np.array([0,1]).reshape(1,2)@robot.g()
        robot.b1_hard[1] = -np.array([0,1]).reshape(1,2)@robot.g()@u_d.value + beta_list[0]*h2 - np.array([0,1]).reshape(1,2)@robot.f()

        for j in range(0,len(obstacle_list)):
            obs_x_r = obstacle_list[j,:].reshape(2,1)
            h_obs, dh_obs_dx = robot.static_safe_set(obs_x_r,0.2) 
            h_obs = -h_obs
            dh_obs_dx = -dh_obs_dx
            robot.A1_hard[j+2,:] = -dh_obs_dx@robot.g()
            robot.b1_hard[j+2] = dh_obs_dx@robot.f() + beta_list[0]*h_obs + dh_obs_dx@robot.g()@u_d.value

        A1_clf.value = robot.A1_soft
        b1_clf.value = robot.b1_soft
        A1_soft.value = robot.A1_hard[0,:].reshape(-1,2)
        b1_soft.value = robot.b1_hard[0,:].reshape(-1,1)
        A1_hard.value = robot.A1_hard[1:,:].reshape(-1,2)
        b1_hard.value = robot.b1_hard[1:,:].reshape(-1,1)
        u1_ref.value = robot.nominal_input(x_r)

        try: 
            constrained_controller.solve(solver=cp.GUROBI, reoptimize=True)
            if constrained_controller.status!="optimal" and constrained_controller.status!="optimal_inaccurate":
                if len(ava_id_list) == 0:
                    break
                x_r_id = ava_id_list.pop(0)
                id_updated = True
                delta_t = 0
                continue
        except:
            if len(ava_id_list) == 0:
                break
            x_r_id = ava_id_list.pop(0)
            id_updated = True
            delta_t = 0
            continue

        u_next = u1.value + u_d.value
        robot.step(u_next)
        delta_t += dt
        t_replan += dt
        if t_replan > 5:
            re_plan = True
        #print('u1_value', np.linalg.norm(u1.value))
        #print(x_r_id)
        robot.render_plot()
        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()
        x_list.append(robot.X[0])
        y_list.append(robot.X[1])
        t += dt
        t_list.append(t)

        if ((h1 >= 0) and (delta_t<delta_t_limit)) or ((h1 < 0) and (delta_t>delta_t_limit)):
            if len(ava_id_list)==0:
                break
            if h1 >= 0:
                centroid = centroids[x_r_id]
                r = radii[x_r_id]
                circle = patches.Circle(centroid, radius=r, color='green', zorder=2)
                ax.add_patch(circle)
            x_r_id = ava_id_list.pop(0)
            id_updated = True
            delta_t = 0
        else:
            continue

centroid = centroids[-1]
r = radii[-1]
circle = patches.Circle(centroid, radius=r, color='red', zorder=2)
ax.add_patch(circle)
plt.ioff()
im = ax.scatter(x_list,y_list,cmap='copper',c=t_list, zorder=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('time(s) colorbar')
plt.show()



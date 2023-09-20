import numpy as np
import math
import time
import cvxpy as cp
import copy
import multiprocessing
from queue import PriorityQueue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm
from robot_models.SingleIntegrator2D import *
from Safe_Set_Series import *
from matplotlib.animation import FFMpegWriter
from Trajectory_Model import *
from discretize_helper_scenario_2 import *

plt.rcParams.update({'font.size': 15}) #27
# Sim Parameters                  
dt = 0.1
tf = 60
num_steps = int(tf/dt)

# Define Parameters for CLF and CBF
U_max = 1.0
d_max = 0.6

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

for i in range(0,obstacle_list.shape[0]):
    circle = patches.Circle(obstacle_list[i,:], radius=0.2, color='black', zorder=0)
    ax.add_patch(circle)
ax.axis('equal')

centroids_comb = []
radii_comb = []
alpha_comb = []

comb = [1,1,1,1,1,1,1,0,1]
for i in range(num_points):
    if comb[i] == 1:
        centroids_comb.append(centroids[i,:])
        radii_comb.append(d_max)
        alpha_comb.append(1.0)

centroids_comb = np.array(centroids_comb)
radii_comb = np.array(radii_comb)
alpha_comb = np.array(alpha_comb)
Safe_Set_Series = Safe_Set_Series2D(centroids=centroids_comb,radii=radii_comb,alpha_list=radii_comb)
num_active_points = len(radii_comb)

for i in range(0,centroids_comb.shape[0]):
    if i != centroids_comb.shape[0]-1:
        circle = patches.Circle(centroids_comb[i,:], radius=d_max, color='blue', zorder=0)
    else:
        circle = patches.Circle(centroids_comb[i,:], radius=d_max, color='red', zorder=10)
    ax.add_patch(circle)
#circle = patches.Circle(centroids[4,:], radius=d_max, color='blue', zorder=0)
#ax.add_patch(circle)
#circle = patches.Circle(centroids[5,:], radius=d_max, color='blue', zorder=0)
#ax.add_patch(circle)
circle = patches.Circle(centroids[7,:], radius=d_max, color='blue', zorder=0)
ax.add_patch(circle)

#Define Disturbance
disturbance = True
disturb_max = 1.25*U_max
disturb_std = 1.5
f_max_1 = 1/(disturb_std*math.sqrt(2*math.pi))
f_max_2 = f_max_1/0.5

x_disturb_1 = np.arange(start=-2*disturb_std, stop=2*disturb_std+0.1, step=0.1)
y_disturb_1 = norm.pdf(x_disturb_1, loc=0, scale=disturb_std)/f_max_1 * disturb_max + 3.5
ax.fill_between(x_disturb_1, y_disturb_1, 3.5, alpha=0.2, color='blue')

y_disturb_2 = np.arange(start=-2*(disturb_std*0.5), stop=2*(disturb_std*0.5)+0.1, step=0.1)
x_disturb_2 = norm.pdf(y_disturb_2, loc=0, scale=disturb_std*0.5)/f_max_2 * disturb_max - 0.6
ax.fill_betweenx(y_disturb_2,x_disturb_2,-0.5, alpha=0.2, color='blue')

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
movie_name = 'series_of_safesets_with_large_wind.mp4'


#Define Search Map
control_hash_table = {}
in_control_hash_table = {}
step = 0.1
x_range = np.arange(start=x_min, stop=x_max+step, step=step)
y_range = np.arange(start=y_min, stop=y_max-0.2+step, step=step)
feasible_candidates = []

for x in x_range:
    for y in y_range:
        if x > -2.9 and x < -0.7 and y < 1.2 and y > 0.8:
            continue
        if x > 0.8 and x < 1.2 and y < 1.3 and y > -0.9:
            continue
        x0 = np.array([x,y])
        feasible_candidates.append(x0)


with multiprocessing.Pool() as pool:
    for (x0_key, forward_set, dist_ford) in pool.map(discretize_u_forward_cal,feasible_candidates):
        for idx, forward_cell in enumerate(forward_set):
            x = ""
            for i in range(len(forward_cell)):
                a = forward_cell[i]
                if a!=',':
                    x += a
                else:
                    break
            y = forward_cell[i+1:]
            x = x_range[int(x)]
            y = y_range[int(y)]
            if y > y_max-0.2 or y < y_min or x > x_max or x < x_min:
                continue
            if x > -2.9 and x < -0.7 and y < 1.2 and y > 0.8:
                continue
            if x > 0.8 and x < 1.2 and y < 1.3 and y > -0.9:
                continue
            if (in_control_hash_table.get(forward_cell)==None):
                backward_set = np.array([x0_key])
                dist_back = np.array([dist_ford[idx]])
            else:
                backward_set, dist_back = control_hash_table.get(forward_cell)
                backward_set = np.append(backward_set,np.array([x0_key]))
                dist_back = np.append(dist_back,np.array([dist_ford[idx]]))
            control_hash_table.update({forward_cell: (backward_set, dist_back)})
            in_control_hash_table.update({forward_cell: True})

x0 = np.array([5.0,0.0])
robot = SingleIntegrator2D(x0, dt, ax=ax, id = 0, color='r',palpha=1.0, num_constraints_hard = 0, num_constraints_soft = 0, plot=False)

final_centroids = Safe_Set_Series.centroids[-1,:]
final_target_centroid = np.array([final_centroids]).reshape(2,1)
r = Safe_Set_Series.radii[-1]
x_final_target_range = np.arange(start=final_target_centroid[0]-r,stop=final_target_centroid[0]+r+step,step=step)
y_final_target_range = np.arange(start=final_target_centroid[1]-r,stop=final_target_centroid[1]+r+step,step=step)

success_list = np.array([])
pos_in_success_table = {}
for x in x_final_target_range:
    for y in y_final_target_range:
        if ((x-final_target_centroid[0])**2 + (y-final_target_centroid[1])**2) <= r**2:
            target_pos = np.array([x,y]).reshape(2,1)
            target_pos_key = str(int((target_pos[0]-x_min)/step))+","+str(int((target_pos[1]-y_min)/step))
            success_list = np.append(success_list,np.array([target_pos_key]))
            pos_in_success_table.update({target_pos_key: True})

while success_list.size > 0:
    current = success_list[0]
    success_list = np.delete(success_list, obj=0, axis=0)
    print(success_list.size)
    if (in_control_hash_table.get(current)==None):
        continue
    else:
        backward_set, _ = control_hash_table.get(current)
    filtered_backward_set = None
    for i in range(backward_set.size):
        has_been_pushed = pos_in_success_table.get(backward_set[i])
        if has_been_pushed==None:
            none_list = np.array([filtered_backward_set == None]).reshape(-1,).tolist()
            if any(none_list):
                filtered_backward_set = np.array([backward_set[i]])
            else:
                filtered_backward_set = np.append(filtered_backward_set,np.array([backward_set[i]]),axis=0)                
            pos_in_success_table.update({backward_set[i]: True})

    none_list = np.array([filtered_backward_set == None]).reshape(-1,).tolist()
    if any(none_list):
        continue
    if len(success_list)> 0:
        success_list = np.append(success_list,filtered_backward_set,axis=0)
    else:
        success_list = filtered_backward_set

x_success_list = []
y_success_list = []
for i, pos in enumerate(pos_in_success_table):
    current = pos
    x = ""
    for i in range(len(current)):
        a = current[i]
        if a!=',':
            x += a
        else:
            break
    y = current[i+1:]
    x = x_range[int(x)] 
    y = y_range[int(y)]
    x_success_list.append(x)
    y_success_list.append(y)

reward = 0
active_safe_set_id = 0
final_path = []
chosen_node = str(int((x0[0]-x_min)/step))+","+str(int((x0[1]-y_min)/step))

delta_t_limit = float(tf)/len(radii_comb)
delta_t = 0
max_iter = 1e5
x_list = []
y_list = []
t_list = []
t = 0
with writer.saving(fig, movie_name, 100): 
    for i in range(num_steps):

        current_pos = robot.X
        current_pos_key = chosen_node

        if Safe_Set_Series.id != active_safe_set_id:
            Safe_Set_Series.id = active_safe_set_id
            centroid = Safe_Set_Series.return_centroid(Safe_Set_Series.id)
            r = Safe_Set_Series.return_radius(Safe_Set_Series.id)
            circle = patches.Circle(centroid, radius=r, color='blue', zorder=1)
            ax.add_patch(circle)
        
        centroid = Safe_Set_Series.return_centroid(Safe_Set_Series.id)
        r = Safe_Set_Series.return_radius(Safe_Set_Series.id)
        
        if len(final_path) == 0:
            possible_node_list = PriorityQueue()
            in_path_list = {}
            x_cent_range = np.arange(start=centroid[0]-r,stop=centroid[0]+r+step,step=step)
            y_cent_range = np.arange(start=centroid[1]-r,stop=centroid[1]+r+step,step=step)
            for x in x_cent_range:
                for y in y_cent_range:
                    if ((x-centroid[0])**2+(y-centroid[1])**2) <= r**2:
                        pos_key = str(int((x-x_min)/step))+","+str(int((y-y_min)/step))
                        in_success_table = pos_in_success_table.get(pos_key)
                        in_control_table = in_control_hash_table.get(pos_key)
                        if (in_success_table) and in_control_table:
                            possible_node_list.put((0, [pos_key]))
                            in_path_list.update({pos_key: True})

            if possible_node_list.empty():
                active_safe_set_id += 1
                continue
            iter_i = 0
            while ~possible_node_list.empty():
                iter_i += 1
                prev_weight, possible_path = possible_node_list.get()
                node = possible_path[-1]

                if node == current_pos_key:
                    final_path = possible_path
                    final_path.pop(-1)
                    break
                
                if iter_i >= max_iter:
                    final_path = []
                    print('too long')
                    break

                if (in_control_hash_table.get(node)==None):
                    continue
                else:
                    backward_set,dist_back = control_hash_table.get(node)

                for idx,cell in enumerate(backward_set):
                    new_path = copy.deepcopy(possible_path)
                    new_path.append(cell)
                    weight = dist_back[idx] + prev_weight
                    if in_path_list.get(cell) != None:
                        curr_weight = in_path_list.get(cell)
                        if (weight < curr_weight):
                            possible_node_list.put((weight, new_path))
                        else:
                            continue
                    else:
                        possible_node_list.put((weight, new_path))
                    in_path_list.update({cell: weight})

        if final_path == []:
            if active_safe_set_id < num_active_points-1:
                active_safe_set_id += 1
            else:
                break
            continue
        chosen_node = final_path.pop(-1)
        chosen_x = ""
        for i in range(len(chosen_node)):
            a = chosen_node[i]
            if a!=',':
                chosen_x += a
            else:
                break
        chosen_y = chosen_node[i+1:]
        chosen_x = x_range[int(chosen_x)] 
        chosen_y = y_range[int(chosen_y)]
        robot.X = np.array([chosen_x,chosen_y]).reshape(-1,1)
        x_list.append(chosen_x)
        y_list.append(chosen_y)
        #robot.render_plot()
        #fig.canvas.draw()
        #fig.canvas.flush_events()
        #print(active_safe_set_id)
        delta_t += dt
        t += dt
        t_list.append(t)


        if (delta_t>delta_t_limit):
            if active_safe_set_id < num_active_points-1:
                active_safe_set_id += 1
                final_path = []
                delta_t = 0
                continue
            else:
                break
            

        if len(final_path)==0:
            centroid = Safe_Set_Series.return_centroid(Safe_Set_Series.id)
            r = Safe_Set_Series.return_radius(Safe_Set_Series.id)
            circle = patches.Circle(centroid, radius=r, color='green', zorder=2)
            ax.add_patch(circle)
            reward += 1
            print("ID: ", Safe_Set_Series.id)
            delta_t = 0
            if active_safe_set_id < num_active_points-1:
                active_safe_set_id += 1
            else:
                break
                
        
        writer.grab_frame()


print(len(pos_in_success_table))
#plt.plot(x_success_list,y_success_list,'b.')
print(len(x_success_list))

plt.ioff()
im = ax.scatter(x_list,y_list,cmap='copper',c=t_list, zorder=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('time(s) colorbar')
plt.show()



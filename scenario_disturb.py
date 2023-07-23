import numpy as np
from scipy.stats import norm



def disturb_value(robot, disturbance, disturb_std, disturb_max, f_max_1, f_max_2, scenario_num):

    if scenario_num == 2:
        if disturbance and robot.X[1]>3.5 and robot.X[0] > -2*disturb_std and robot.X[0] < 2*disturb_std:
            y_disturb = norm.pdf(robot.X[0], loc=0, scale=disturb_std)[0]/f_max_1 * disturb_max
            x_disturb = 0.0
        elif disturbance and robot.X[0]>-0.5 and robot.X[0] < 1.8\
            and robot.X[1] > -2*(disturb_std*0.5) and robot.X[1] < 2*(disturb_std*0.5):
            x_disturb = norm.pdf(robot.X[1], loc=0, scale=disturb_std*0.5)[0]/f_max_2 * disturb_max
            y_disturb = 0.0
        else:
            x_disturb = 0.0
            y_disturb = 0.0
    else:
        y_disturb = 0.0
        x_disturb = 0.0

    u_d_value = np.array([x_disturb, y_disturb]).reshape(2,1)
    
    return u_d_value
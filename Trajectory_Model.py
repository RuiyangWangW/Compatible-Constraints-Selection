import numpy as np
import math
from utils.utils import wrap_angle


class Trajectory2D:
    def __init__(self, trajectory_points, tot_time, poly_degree, type='SingleIntegrator'):
        self.type = type
        self.trajectory_points = trajectory_points
        self.num_points = trajectory_points.shape[0]
        self.time_step = tot_time/float(self.num_points-1)
        self.time_list = np.arange(start=0,stop=tot_time+self.time_step,step=self.time_step)
        self.poly_fit_x = np.polyfit(self.time_list, self.trajectory_points[:,0], deg=poly_degree)
        self.poly_fit_y = np.polyfit(self.time_list, self.trajectory_points[:,1], deg=poly_degree)
        if type == 'Unicycle':
            self.poly_fit_theta = np.polyfit(self.time_list, self.trajectory_points[:,3], deg=poly_degree)
        self.tot_time = tot_time


    def get_current_target(self, time):
        if(time>self.tot_time):
            return self.trajectory_points[-1,:].reshape(-1,1)
        
        target = np.zeros((4,1))
        target[0] = np.polyval(self.poly_fit_x, time)
        target[1] = np.polyval(self.poly_fit_y, time)
        if self.type == 'Unicycle':
            target[3] = np.polyval(self.poly_fit_theta, time)
            return target
        else:
            return target[0:2].reshape(-1,1)

    def x_r_dot(self, time):
        dx_r_dt = np.zeros((4,1))
        poly_dx_dt = np.polyder(self.poly_fit_x)
        dx_r_dt[0] = np.polyval(poly_dx_dt, time)
        poly_dy_dt = np.polyder(self.poly_fit_y)
        dx_r_dt[1] = np.polyval(poly_dy_dt, time)
        if self.type == 'Unicycle':
            poly_dtheta_dt = np.polyder(self.poly_fit_theta)
            dx_r_dt[3] == np.polyval(poly_dtheta_dt, time)
            if time >= self.tot_time:
                return np.zeros((4,1))
            else:
                return dx_r_dt
        else:
            if time >= self.tot_time:
                return np.zeros((2,1))
            else:
                return dx_r_dt[:2].reshape((2,1))


def PointsInCircum(r,n=100):
    Points = [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]
    return np.array(Points)


def PointsInCircum_with_theta(r,n=100):
    Points = [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r, 0, wrap_angle(2*math.pi/n*x+math.pi/2)) for x in range(0,n+1)]
    return np.array(Points)
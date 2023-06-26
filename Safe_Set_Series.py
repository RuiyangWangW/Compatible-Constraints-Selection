import numpy as np
from robot_models.SingleIntegrator2D import *


class Safe_Set_Series2D:
    
    def __init__(self,centroids,radii,alpha_list):
        self.centroids = centroids
        self.radii = radii
        self.alpha_list = alpha_list
        self.id = None
        self.num_sets = len(centroids)

    def safe_set_h(self,robot):
        h, dh_dx = robot.static_safe_set(target=self.centroids[self.id,:].reshape(2,1),d_max=self.radii[self.id])
        return h, dh_dx
    
    def return_centroid(self,id):
        return self.centroids[id]
    
    def return_radius(self,id):
        return self.radii[id]

    def safe_set_constraints(self,robot,u_d):
        h, dh_dx = robot.static_safe_set(target=self.centroids[self.id,:].reshape(2,1),d_max=self.radii[self.id])
        print("h =", h)
        A = -dh_dx@robot.g()
        b = -dh_dx@robot.f() + self.alpha_list[self.id]*h + dh_dx@robot.g()@u_d.value
        return A, b
    
    def sets_reached(self,robot):
        h, _ = robot.static_safe_set(target=self.centroids[self.id,:].reshape(2,1),d_max=self.radii[self.id]) 
        if h >= 0.2:
            return True
        else:
            return False
    
    def update_targets(self, robot, failed):
        if self.sets_reached(robot) or failed:
            if self.id < self.num_sets-1:
                self.id += 1
    
        


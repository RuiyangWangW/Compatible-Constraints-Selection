import numpy as np
from robot_models.SingleIntegrator2D import *


class Safe_Set_Series2D:
    
    def __init__(self,centroids,radii):
        self.centroids = centroids
        self.radii = radii
        self.id = None
        self.num_sets = len(centroids)

    def return_centroid(self,id):
        return self.centroids[id]
    
    def return_radius(self,id):
        return self.radii[id]

    
        


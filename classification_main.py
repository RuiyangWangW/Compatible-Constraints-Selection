import numpy as np
import cvxpy as cp
import random
import matplotlib.pyplot as plt
import time
from chinneck import *
from lagrange_score import *
from optimal_classify import *
from sklearn.svm import LinearSVC

for k in range (1,11,1):
    #Define Random Seed
    random.seed(k)

    #Define Num of Random Points
    n_points = 100
    n_outlier = 10

    #Define Optimal Classification Line in the form ax+b
    a = 0.8
    b = 2.3

    x_class_0 = []
    y_class_0 = []
    x_class_1 = []
    y_class_1 = []

    #Generate Random Points
    for i in range(n_points-n_outlier):
        x = random.uniform(0,10)
        y = random.uniform(0,10)
        if (a*x + b) >= y:
            x_class_0.append(x)
            y_class_0.append(y)
        else:
            x_class_1.append(x)
            y_class_1.append(y)

    for i in range(n_outlier):
        x = random.uniform(0,10)
        y = random.uniform(0,10)
        if (a*x + b) >= y:
            x_class_1.append(x)
            y_class_1.append(y)
        else:
            x_class_0.append(x)
            y_class_0.append(y)

    #Plot 

    #plt.plot(x_class_0, y_class_0, 'b^')
    #plt.plot(x_class_1, y_class_1, 'ro')
    #x_class = np.hstack([np.array(x_class_0),np.array(x_class_1)]).reshape(-1,1)
    #y_class = np.hstack([np.array(y_class_0),np.array(y_class_1)]).reshape(-1,1)
    #linsvc = LinearSVC(C=1.0)
    #linsvc.fit(X=np.hstack([x_class,y_class]).reshape(-1,2),
    #           y=np.hstack([np.zeros((len(x_class_0),)),np.ones((len(x_class_1),))]).reshape(-1,))
    #x_min, x_max = 0, 10
    #y_min, y_max = 0, 10
    #xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
    #                     np.arange(y_min, y_max, 0.01))


    #Z = linsvc.predict(np.c_[xx.ravel(), yy.ravel()])

    ## Put the result into a color plot
    #Z = Z.reshape(xx.shape)
    """
    start_time = time.time()
    result = optimal_classification(x_class_0,y_class_0,x_class_1,y_class_1)
    end_time = time.time()
    optimal_a = result["optimal_a"]
    optimal_b = result["optimal_b"]
    cover_set_0 = result["cover_set_0"]
    cover_set_1 = result["cover_set_1"]
    print(k,"th test")
    print("Time Elapsed: ", end_time-start_time)
    print("Num of constraints removed: ", len(cover_set_0)+len(cover_set_1), "from optimal")
    """
    
    start_time = time.time()
    result = chinneck_1(x_class_0,y_class_0,x_class_1,y_class_1)
    end_time = time.time()
    #optimal_a = result["optimal_a"]
    #optimal_b = result["optimal_b"]
    cover_set_0 = result["cover_set_0"]
    cover_set_1 = result["cover_set_1"]
    print(k,"th test")
    print("Time Elapsed: ", end_time-start_time)
    print("Num of constraints removed: ", len(cover_set_0)+len(cover_set_1), "from chinneck_2")

    """
    start_time = time.time()
    result = lagrange_score_1(x_class_0,y_class_0,x_class_1,y_class_1)
    end_time = time.time()
    #optimal_a = result["optimal_a"]
    #optimal_b = result["optimal_b"]
    cover_set_0 = result["cover_set_0"]
    cover_set_1 = result["cover_set_1"]
    print(k,"th test")
    print("Time Elapsed: ", end_time-start_time)
    print("Num of constraints removed: ", len(cover_set_0)+len(cover_set_1), "from lagrange_1")

    start_time = time.time()
    result = lagrange_score_2(x_class_0,y_class_0,x_class_1,y_class_1)
    end_time = time.time()
    #optimal_a = result["optimal_a"]
    #optimal_b = result["optimal_b"]
    cover_set_0 = result["cover_set_0"]
    cover_set_1 = result["cover_set_1"]
    print(k,"th test")
    print("Time Elapsed: ", end_time-start_time)
    print("Num of constraints removed: ", len(cover_set_0)+len(cover_set_1), "from lagrange_2")
    """

    #plt.plot([0,10], [optimal_b,10*optimal_a+optimal_b],'g-', linewidth=4)
    #plt.xlim((0,10))
    #plt.ylim((0,10))
    #plt.legend(['class 0', 'class 1','output'])
    #plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #plt.show()

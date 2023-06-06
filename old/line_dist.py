import numpy as np
import torch
import torch.nn as nn
import math
import random

import matplotlib.pyplot as plt
import pickle

def get_dist_point_line(l, p):
    # l: (m, a, b)
    # p: (x1, y1)
    m, a, b = l
    x1, y1 = p
    return abs(m*x1 - y1 - m*a + b) / math.sqrt(m*m + 1)

def get_ab(R, m):
    sin = np.sqrt(1/(m**2 + 1))
    cos = -m * sin
    y1 = R*sin
    # y2 = -y1
    x1 = R*cos
    # x2 = -x1
    # y = -(1/m)(x-x1) + y1
    random_x = random.uniform(-x1, x1)
    y = -(1/m)*(random_x-x1) + y1

    return random_x, y

def get_e_v_c(conv_size=(7,7), scale=1, num=1000000, eps=1e-8):
    px = [i for i in range(conv_size[0])] * conv_size[1]
    py = [i for i in range(conv_size[1], 0, -1)] * conv_size[0]
    mat_x = torch.tensor(px, dtype=torch.float64).view(conv_size[0], conv_size[1])
    mat_y = torch.tensor(py, dtype=torch.float64).view(conv_size[1], conv_size[0]).transpose(0,1)
    mat_x = mat_x +0.5 - conv_size[0] / 2
    mat_y = mat_y -0.5 - conv_size[1] / 2
    print(mat_x)
    # mat_size = mat.shape # (Filters, Channels, W, H)
    # print(mat_x)
    print(mat_y)
    domain = conv_size[0] * scale

    summed = torch.zeros(conv_size)
    squared_summed = torch.zeros(conv_size)
    summed_XY = torch.zeros((conv_size[0]**2, conv_size[1]**2))

    # n_in = mat_size[2] * mat_size[3] * mat_size[1]
    # n_in = 9
    binary = [-1, 1]
    beta = 0.7
    R = beta + 2*np.sqrt(2)
    var_w = beta**3 / (3 * R) 
    tmp_cov = beta**4 / (4*R**2)
    # print((np.pi * beta**2) / (8*beta + 4 + np.pi*beta**2))
    print(var_w)
    print(1-beta/R) # prob of 0
    print(num * (1-beta/R) + (num/10) * beta/R) # number of 0 ~ 0.1
    print(f'tmp cov: {tmp_cov}')
    impact = 1
    dist_lst = []
    for n in range(num):
        theta = random.uniform(0, np.pi)
        m = np.tan(theta)
        a, b = get_ab(R, m)
        sign = random.choice(binary)
        # distance = torch.abs((m * mat_x - mat_y - m * a + b)) / np.sqrt(m*m+1)
        distance = torch.abs((m * mat_x - mat_y - m * a + b)) / np.sqrt(m*m+1)
        # distance[distance > np.sqrt(2)*0.5] = 0
        distance = beta - distance
        distance[distance < 0] = 0
        distance = distance * sign
        dist_lst.append(distance[0][0].item())
        summed += distance
        squared_summed += distance**2
        summed_XY += distance.view(-1, 1) * distance.view(1, -1)

    zs = 0
    for i in dist_lst:
        if i == 0 : zs += 1
    plt.hist(dist_lst, bins=9)
    plt.show()

    E = (summed / num)
    V = squared_summed / num - (summed / num) ** 2
    # print(f'mean is {E}')
    # print(f'var is {V}')
    mean_mean = E.view(-1, 1) * E.view(1, -1)
    cov = summed_XY / num
    # cov = torch.tensor([[0.0555, 0.0214, 0.0096, 0.0214, 0.0140, 0.0086, 0.0096, 0.0086, 0.0067],
    #                     [0.0214, 0.0555, 0.0214, 0.0140, 0.0214, 0.0140, 0.0086, 0.0096, 0.0086],
    #                     [0.0096, 0.0214, 0.0555, 0.0086, 0.0140, 0.0214, 0.0067, 0.0086, 0.0096],
    #                     [0.0214, 0.0140, 0.0086, 0.0555, 0.0214, 0.0096, 0.0214, 0.0140, 0.0086],
    #                     [0.0140, 0.0214, 0.0140, 0.0214, 0.0555, 0.0214, 0.0140, 0.0214, 0.0140],
    #                     [0.0086, 0.0140, 0.0214, 0.0096, 0.0214, 0.0555, 0.0086, 0.0140, 0.0214],
    #                     [0.0096, 0.0086, 0.0067, 0.0214, 0.0140, 0.0086, 0.0555, 0.0214, 0.0096],
    #                     [0.0086, 0.0096, 0.0086, 0.0140, 0.0214, 0.0140, 0.0214, 0.0555, 0.0214],
    #                     [0.0067, 0.0086, 0.0096, 0.0086, 0.0140, 0.0214, 0.0096, 0.0214, 0.0555]])
    print(E)
    print(V)
    print('cov')
    print(cov)
    for i in cov:
        print(i)
    # print(cov)
    
    U,S,V = np.linalg.svd(cov)
    epsilon = 0
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]

    # print('U')
    # print(U)
    # print('l')
    # print(np.diag(1.0/np.sqrt(S + epsilon)))
    # print('l no e')
    # print(np.diag(1.0/np.sqrt(S)))
    # print(ZCAMatrix)
    
    original = torch.zeros(9, 1)
    transformed = torch.zeros(9, 1)
    for i in range(3):
        # theta = 0
        # for t in range(8):
        theta = random.uniform(0, np.pi)
        # theta = (theta + np.pi / 4 ) % np.pi
        m = np.tan(theta)
        a, b = get_ab(R, m)
        sign = random.choice(binary)
        # distance = torch.abs((m * mat_x - mat_y - m * a + b)) / np.sqrt(m*m+1)
        distance = torch.abs((m * mat_x - mat_y - m * a + b)) / np.sqrt(m*m+1)
        # distance[distance > np.sqrt(2)*0.5] = 0
        distance = beta - distance
        distance[distance < 0] = 0
        distance = distance * sign
        
        distance = distance.view(-1, 1)
        original += distance
        # print(distance)
        
        xZCAMatrix = np.dot(ZCAMatrix, distance)
        transformed += xZCAMatrix
        # print(xZCAMatrix)
    # print(original)
    # print(transformed)

E_V_C_dict = {}
lst = [(5, 5, 1)]

for i in lst:
    # E, V, C = get_e_v_c((i[0], i[1]), i[2], 100000000) #1억
    # E, V, C = get_e_v_c((i[0], i[1]), i[2], 10000000) #1천
    get_e_v_c((i[0], i[1]), i[2], 1000000) #1백만
    # get_e_v_c((i[0], i[1]), i[2], 100000) #십만
    # get_e_v_c((i[0], i[1]), i[2], 10000) #1만
    # E, V, C = get_e_v_c((i[0], i[1]), i[2], 10)

# with open('EVC3.txt', 'wb') as evc_f:
#     pickle.dump(E_V_C_dict, evc_f)
from dis import dis
import torch
import numpy as np
import torch
import torch.nn as nn
import math
import random
import utils
import os
import pickle

class ReprInit:
    def __init__(self, model, args, **kwargs):
        self.model = model
        self.args = args
        # self.repr_init_rate = kwargs['repr_init_rate']
        self.betas = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
        self.weight_points = {}
        self.weight_var = {}
        self.zca_dict = {}
        self.min_line = args.min_line
        self.max_line = args.max_line

    def apply_initialization(self):
        ### init fc and bn layers
        # self.model.apply(utils.weights_init_fan_out_linear)
        # self.model.apply(utils.weights_init_batchnorm)

        ### init conv layers
        ## TODO: idx wrong and how to deal with odwnsample??? and 1x1 conv???

        # idx = 0
        # for name, layer in self.model.named_modules():
        #     if isinstance(layer, nn.Conv2d):
        #         if not ('downsample' in name) and self.chk_1x1(layer):
        #             if idx < len(self.repr_init_rate) and self.repr_init_rate[idx] > 0:
        #                 print(name + ' repr_init')
        #                 self.repr_init_weight(layer, idx, self.repr_init_rate[idx])
        #             else:
        #                 print(name + ' init')
        #                 if self.args.fan_in:
        #                     nn.init.kaiming_normal_(layer.weight, mode='fan_in')
        #                 else:
        #                     nn.init.kaiming_normal_(layer.weight, mode='fan_in')
        #                 if layer.bias is not None:
        #                     layer.bias.data.fill_(0)
        #             idx += 1

        idx = 0
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                if self.chk_1x1(layer):
                    # nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                    # if layer.bias is not None:
                    #     layer.bias.data.fill_(0)
                    pass
                else:
                    print(layer)
                    self.repr_init_weight(layer, idx)
                    idx += 1

    # get E, V, Cov, ZCA matrix
    def prepare_init(self, conv_size, epsilon=0):
        W, H = conv_size
        
        px = [i for i in range(conv_size[0])] * conv_size[1]
        py = [i for i in range(conv_size[1], 0, -1)] * conv_size[0]
        mat_x = torch.tensor(px, dtype=torch.float32).view(conv_size[0], conv_size[1])
        mat_y = torch.tensor(py, dtype=torch.float32).view(conv_size[1], conv_size[0]).transpose(0,1)
        mat_x = mat_x + 0.5 - conv_size[0] / 2
        mat_y = mat_y - 0.5 - conv_size[1] / 2

        self.weight_points[W] = []
        self.weight_points[W].append(mat_x)
        self.weight_points[W].append(mat_y)
        
        center_point = (int((conv_size[0] - 1) / 2), int((conv_size[1] - 1) / 2))
        # d_c
        R = np.sqrt((mat_x[center_point[0]][center_point[1]] - mat_x[0][0])**2 
                                + (mat_y[center_point[0]][center_point[1]] - mat_y[0][0])**2)

        self.weight_var[W] = {}
        self.zca_dict[W] = {}
        for b in self.betas:
            cov = torch.zeros((W*H, W*H), dtype=torch.float32)
            var_w = b**3 / (3 * R)
            self.weight_var[W][b] = var_w
            for i in range(W*H):
                cov[i][i] = var_w

            d_cov_dict = {}
            for i in range(W*H):
                for j in range(i+1, W*H):
                    x1 = i % W
                    y1 = i // W
                    x2 = j % W
                    y2 = j // W

                    d = math.sqrt((mat_x[x1][y1] - mat_x[x2][y2])**2 + (mat_y[x1][y1] - mat_y[x2][y2])**2)
                    if d in d_cov_dict.keys():
                        cov_ = d_cov_dict[d]
                    else:
                        # i)
                        if d >= b and d <= 2*b:
                            f1 = ((12*d**3-54*b*d**2+24*b**3)*math.asin((d-b)/d)+(-4*d**2-38*b**2)*math.sqrt(d**2-b**2)+(4-6*math.pi)*d**3+math.sqrt(2*b*d-b**2)*(12*d**2-50*b*d-2*b**2)+(27*math.pi*b-18*b*math.asin(b/d))*d**2+36*b**2*d-24*b**3*math.asin(b/d))/36
                            f2 = -((6*d**3-27*b*d**2+12*b**3)*math.asin((d-b)/d)-3*math.pi*d**3+math.sqrt(2*b*d-b**2)*(6*d**2-25*b*d-b**2)+math.sqrt(d**2-b**2)*(2*d**2+19*b**2)+(9*b*math.asin(b/d)+9*math.pi*b)*d**2+12*b**3*math.asin(b/d)-12*math.pi*b**3)/18
                            f3 = -(-2*d**3+math.sqrt(d**2-b**2)*(2*d**2-8*b**2)+9*b**2*d-6*b**3*math.asin(b/d))/9 # b, 0, y-x
                            cov_ = (f1+f2+f3) / (math.pi*R)


                        # # ii)
                        elif d >= 2*b:
                            f1 = -((-2*d**2-22*b**2)*math.sqrt(d**2-4*b**2)-2*d**3+math.sqrt(d**2-b**2)*(4*d**2+38*b**2)+(18*b*math.asin(b/d)-9*b*math.asin((2*b)/d))*d**2-18*b**2*d-24*b**3*math.asin((2*b)/d)+24*b**3*math.asin(b/d))/18 # b, 0 2b-x-y
                            f2 = -(-2*d**3+math.sqrt(d**2-b**2)*(2*d**2-8*b**2)+9*b**2*d-6*b**3*math.asin(b/d))/9 # b, 0, y-x
                            cov_ = (f1+f2)/(math.pi*R)

                        # iii)
                        # d<=b/2
                        elif d<=b/2:
                            f1 = (d*((3*math.pi+1)*d**2-9*math.pi*b*d+(9*math.pi-9)*b**2))/9
                            f2 = (d*(4*d**2-9*math.pi*b*d+36*b**2))/36
                            f3 = -(math.pi*(2*d**3-3*b*d**2+3*b**2*d-b**3))/3
                            f4 = (5*math.pi*d**2)/8
                            cov_ = (f1+f2+f3+f4) / (math.pi*R)


                        # iv)
                        # b/2<= d <= b
                        elif d >= b/2 and d <= b:
                            f1 = ((6*d**3-18*b*d**2+18*b**2*d-6*b**3)*math.asin((d-b)/d)+(2-3*math.pi)*d**3+math.sqrt(2*b*d-b**2)*(6*d**2-16*b*d+8*b**2)+9*math.pi*b*d**2-9*math.pi*b**2*d+3*math.pi*b**3)/18
                            f2 = -((12*d**3-18*b*d**2+18*b**2*d-12*b**3)*math.asin((d-b)/d)-6*math.pi*d**3+math.sqrt(2*b*d-b**2)*(12*d**2-14*b*d+16*b**2)+9*math.pi*b*d**2-9*math.pi*b**2*d)/18
                            f3 = ((6*d**3-6*b**3)*math.asin((d-b)/d)+(2-3*math.pi)*d**3+math.sqrt(2*b*d-b**2)*(6*d**2+2*b*d+8*b**2)-18*b**2*d+3*math.pi*b**3)/18
                            f4 = (d*(4*d**2-9*math.pi*b*d+36*b**2))/36
                            cov_ = (f1+f2+f3+f4) / (math.pi*R)
                        d_cov_dict[d] = cov_
                    cov[i][j] = cov_
                    
            for i in range(W*H):
                for j in range(i):
                    cov[j][i] = cov[i][j]

            U,S,V = np.linalg.svd(cov)
            # ZCA Whitening matrix: U * Lambda * U'
            ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
            ZCAMatrix = torch.tensor(ZCAMatrix, dtype=torch.float32)
        
            self.zca_dict[W][b] = ZCAMatrix
        # return mat_x, mat_y, ZCAMatrix

    def chk_1x1(self, layer):
        mat = layer.weight.data
        mat_size = mat.shape # (Filters, Channels, W, H)
        conv_size = (mat_size[2], mat_size[3])

        if conv_size[0] == 1 and conv_size[1] == 1:
            return True
        else:
            return False

    def repr_init_weight(self, layer, idx):
        with torch.no_grad():
            mat = layer.weight.data
            if layer.bias is not None:
                layer.bias.data.fill_(0)
            mat_size = mat.shape # (Filters, Channels, W, H)
            conv_size = (mat_size[2], mat_size[3])

            if conv_size[0] in self.weight_points:
                pass
            else:
                self.prepare_init(conv_size, 0)

            mat_x = self.weight_points[conv_size[0]][0]
            mat_y = self.weight_points[conv_size[0]][1]
            ZCAMatrix = self.zca_dict[conv_size[0]]
            weight_var = self.weight_var[conv_size[0]]
            
            mat.fill_(0)

            center_point = (int((conv_size[0] - 1) / 2), int((conv_size[1] - 1) / 2))
            # d_c
            distance_center = np.sqrt((mat_x[center_point[0]][center_point[1]] - mat_x[0][0])**2 
                                    + (mat_y[center_point[0]][center_point[1]] - mat_y[0][0])**2)

            binary = [-1, 1]
            
            beta = random.choice(self.betas)
            R = beta + distance_center

            impact = float(self.args.impact)

            # n_in = mat_size[2] * mat_size[3] * mat_size[1]
            n_in, n_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)


            # if no zca
            # if self.args.first_conv:
            #     min_lin = round(self.min_line * (conv_size[0] / 3))
            #     max_lin = round(self.max_line * (conv_size[0] / 3))
            #     for f in range(mat_size[0]):
            #         # repr_prob_rand = random.uniform(0, 1)
            #         # if repr_prob >= repr_prob_rand:
                    
            #         for i in range(num_lin):
            #             num_lin = random.randint(min_lin, max_lin)
            #             theta = random.uniform(0, math.pi)
            #             m = np.tan(theta)
            #             a, b = utils.get_ab(R, m)
            #             sign = random.choice(binary)

            #             distance = torch.abs((m * mat_x - mat_y - m * a + b)) / np.sqrt(m*m+1)
            #             distance = beta - distance
            #             distance[distance < 0] = 0
            #             distance = distance * sign * np.sqrt(impact) * np.sqrt(2/n_out) / np.sqrt(num_lin) / np.sqrt(weight_var[beta])

            #             for c in range(mat_size[1]):
            #                 mat[f][c] += distance

            #         mat[f] += torch.normal(0, np.sqrt(1-impact) * np.sqrt(2/n_out), size=(mat_size[1], mat_size[2], mat_size[3]))
            
            # if first conv layer
            if self.args.first_conv:
                if idx == 0:
                    min_lin = round(self.min_line * (conv_size[0] / 3))
                    max_lin = round(self.max_line * (conv_size[0] / 3))
                    for f in range(mat_size[0]):
                        # repr_prob_rand = random.uniform(0, 1)
                        # if repr_prob >= repr_prob_rand:
                        num_lin = random.randint(min_lin, max_lin)
                        dist_lst = []
                        for i in range(num_lin):
                            theta = random.uniform(0, math.pi)
                            m = np.tan(theta)
                            a, b = utils.get_ab(R, m)
                            sign = random.choice(binary)

                            distance = torch.abs((m * mat_x - mat_y - m * a + b)) / np.sqrt(m*m+1)
                            distance = beta - distance
                            distance[distance < 0] = 0
                            distance = distance * sign
                            distance = distance.view(-1, 1)
                            dist_lst.append(distance)

                            for d in dist_lst[1:]:
                                dist_lst[0] += d

                            if self.args.no_zca:
                                for c in range(mat_size[1]):
                                    mat[f][c] = dist_lst[0]
                                    # TODO: variance to fan_out, first conv
                            else:
                                xZCAMatrix = np.sqrt(impact) * np.sqrt(2/n_out) / np.sqrt(num_lin) * torch.mm(ZCAMatrix[beta], dist_lst[0]) 
                                xZCAMatrix = xZCAMatrix.view(conv_size[0], conv_size[1])
                                if not impact == 0:
                                    xZCAMatrix += torch.normal(0, np.sqrt(1-impact) * np.sqrt(2/n_out), size=(mat_size[2], mat_size[3]))
                                
                                for c in range(mat_size[1]):
                                    mat[f][c] = xZCAMatrix
                                

                        # else:
                        #     gaussian_noise = torch.normal(0, np.sqrt(2/n_out), size=(mat_size[1], mat_size[2], mat_size[3]))
                        #     mat[f] += gaussian_noise

                # not first conv layer
                else:
                    min_lin = round(self.min_line * (conv_size[0] / 3))
                    max_lin = round(self.max_line * (conv_size[0] / 3))
                    for f in range(mat_size[0]):
                        # repr_prob_rand = random.uniform(0, 1)
                        # if repr_prob >= repr_prob_rand:
                        
                        for c in range(mat_size[1]):
                            num_lin = random.randint(min_lin, max_lin)
                            dist_lst = []
                            for i in range(num_lin):
                                theta = random.uniform(0, math.pi)
                                m = np.tan(theta)
                                a, b = utils.get_ab(R, m)
                                sign = random.choice(binary)

                                distance = torch.abs((m * mat_x - mat_y - m * a + b)) / np.sqrt(m*m+1)
                                distance = beta - distance
                                distance[distance < 0] = 0
                                distance = distance * sign
                                distance = distance.view(-1, 1)
                                dist_lst.append(distance)

                            for d in dist_lst[1:]:
                                dist_lst[0] += d

                            if self.args.no_zca:
                                mat[f][c] = dist_lst[0]
                                # TODO: variance to fan_out
                            else:
                                xZCAMatrix = np.sqrt(impact) * np.sqrt(2/n_out) / np.sqrt(num_lin) * torch.mm(ZCAMatrix[beta], dist_lst[0]) 
                                xZCAMatrix = xZCAMatrix.view(conv_size[0], conv_size[1])
                                if not impact == 0:
                                    xZCAMatrix += torch.normal(0, np.sqrt(1-impact) * np.sqrt(2/n_out), size=(mat_size[2], mat_size[3]))

                                mat[f][c] = xZCAMatrix

                        # else:
                        #     gaussian_noise = torch.normal(0, np.sqrt(2/n_out), size=(mat_size[1], mat_size[2], mat_size[3]))
                        #     mat[f] += gaussian_noise

            # False self.args.first_conv
            else:
                min_lin = round(self.min_line * (conv_size[0] / 3))
                max_lin = round(self.max_line * (conv_size[0] / 3))
                for f in range(mat_size[0]):
                    repr_prob_rand = random.uniform(0, 1)
                    # if repr_prob >= repr_prob_rand:
                    
                    for c in range(mat_size[1]):
                        num_lin = random.randint(min_lin, max_lin)
                        dist_lst = []
                        for i in range(num_lin):
                            theta = random.uniform(0, math.pi)
                            m = np.tan(theta)
                            a, b = utils.get_ab(R, m)
                            sign = random.choice(binary)

                            distance = torch.abs((m * mat_x - mat_y - m * a + b)) / np.sqrt(m*m+1)
                            distance = beta - distance
                            distance[distance < 0] = 0
                            distance = distance * sign
                            distance = distance.view(-1, 1)
                            dist_lst.append(distance)

                        for d in dist_lst[1:]:
                            dist_lst[0] += d

                        if self.args.no_zca:
                            mat[f][c] = dist_lst[0]
                            # TODO: variance to fan_out

                        else:
                            xZCAMatrix = np.sqrt(impact) * np.sqrt(2/n_out) / np.sqrt(num_lin) * torch.mm(ZCAMatrix[beta], dist_lst[0]) 
                            xZCAMatrix = xZCAMatrix.view(conv_size[0], conv_size[1])
                            if not impact == 0:
                                xZCAMatrix += torch.normal(0, np.sqrt(1-impact) * np.sqrt(2/n_out), size=(mat_size[2], mat_size[3]))

                            mat[f][c] = xZCAMatrix

                    # else:
                    #     gaussian_noise = torch.normal(0, np.sqrt(2/n_out), size=(mat_size[1], mat_size[2], mat_size[3]))
                    #     mat[f] += gaussian_noise
            ################## bates init ###################
            # min_lin = 1
            # max_lin = 2
            # for f in range(mat_size[0]):
            #     theta_tilde = random.uniform(0, np.pi)
            #     n = random.randint(min_lin, max_lin)
            #     bates_n = 3
            #     repr_prob_rand = random.uniform(0, 1)
            #     repr_prob = 1.0
            #     if repr_prob >= repr_prob_rand:
            #     # bates_n = random.randint(1, 4)
            #         # for c in range(mat_size[1]):
            #         for c in range(1):
            #             dist_lst = []
            #             for i in range(n):
            #                 theta = 0
            #                 for b in range(bates_n):
            #                     theta += random.uniform(theta_tilde - (np.pi / 2), theta_tilde + (np.pi / 2)) / bates_n
            #                 m = np.tan(theta)
            #                 a, b = utils.get_ab(R, m)
            #                 sign = random.choice(binary)

            #                 distance = torch.abs((m * mat_x - mat_y - m * a + b)) / np.sqrt(m*m+1)
            #                 distance = beta - distance
            #                 distance[distance < 0] = 0
            #                 distance = distance * sign

            #                 distance = distance.view(-1, 1)
            #                 dist_lst.append(distance)
            #             distance = dist_lst[0]
            #             for i in range(1,n):
            #                 distance += dist_lst[i]
            #             # modified_impact = impact * n / (1 - impact + impact * n)
            #             modified_impact = impact
            #             distance = distance.view(conv_size[0], conv_size[1])
            #             # xZCAMatrix = np.sqrt(modified_impact) * np.sqrt(2/n_out) / np.sqrt(n) * torch.mm(ZCAMatrix, distance) 
            #             # transformed = xZCAMatrix.view(conv_size[0], conv_size[1])
            #             distance = torch.stack([distance, distance, distance], dim=0)

            #             gaussian_noise = torch.normal(0, np.sqrt(1-modified_impact) * np.sqrt(2/n_out) / np.sqrt(var_w), size=(3, conv_size[0], conv_size[1]))
            #             # transformed += gaussian_noise
            #             distance += gaussian_noise
                    
            #             # mat[f][c] += transformed
            #             mat[f] += distance
            #     else:
            #         gaussian_noise = torch.normal(0, np.sqrt(2/n_out), size=(mat_size[1], mat_size[2], mat_size[3]))
            #         mat[f] += gaussian_noise
            
# if __name__ == '__main__':
#     repr_init()
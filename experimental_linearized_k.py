# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:44:02 2022

@author: Stijn
"""
import numpy as np
g=9.81
m1=10
m2=20
m3=30
m4=25

X1=np.array([[0.042],[0.078],[0.063]])
C1=np.array([[m2*g],[m3*g],[m4*g]])
k1=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X1),X1)),np.transpose(X1)),C1)
print(k1)

X2=np.array([[0.093],[0.154]])
C2=np.array([[m2*g],[m3*g]])
k2=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X2),X2)),np.transpose(X2)),C2)
print(k2)


X3=np.array([[0.042]])
C3=np.array([[m2*g]])
k3=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X3),X3)),np.transpose(X3)),C3)
print(k3)
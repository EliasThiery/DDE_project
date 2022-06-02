# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:45:31 2022

@author: Stijn
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
def extractData(filename):
    data = []
    with open(filename,"r") as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            data.append(row)

    data[0]= [float(i) for i in data[0]]
    data[2] = [float(i) for i in data[2]]
    data = [data[0],data[2]]
    return data

data10kg1spring=extractData("10kg_1spring.csv")
#data10kg1spring=extractData("10kg_3springs.csv")
#data10kg1spring=extractData("30kg_1spring.csv")
#data10kg1spring=extractData("30kg_3springs.csv")

#DFT shizzle
Ts = data10kg1spring[0][-1]/len(data10kg1spring[0]) #sampling period
fs = 1/Ts #sampling frequency
def DFT(data):
    data[1]=data[1]-np.mean(data[1]) #center the data
    DFT = np.fft.fft(data) #dft 
    t_fft = np.arange(0,len(data)/2-1,1) #time domain
    f_fft = t_fft*fs/len(data) #x-axis dft, Adrien dumbass carried 23/05/2022
    #freq =  np.fft.fftfreq(t_fft.shape[-1])
    #plt.plot(f_fft,abs(DFT[0:int(len(DFT)/2)])*2) #maal 2 is mogelijks bs
    #plt.show
    index = 0
    High = np.max(abs(DFT[1:int(len(DFT)/2)])*2) #find max amplitude
    for i in range(1,round(len(DFT)/2)): 
        if High == abs(DFT[i])*2:
            index = i
    eigenfrequentie = f_fft[index] 
    return eigenfrequentie

#eigenomega = 2*np.pi*eigenfrequentie
#k = eigenomega**2*10
#print(k)

data = []
with open("10kg_1spring.csv","r") as file:
    reader = csv.reader(file, delimiter=',')
    row_num = 0
    for row in reader:
        if row_num == 2:
            temp = row
        row_num += 1
    temp = [float(i) for i in temp]
    plt.plot(temp[210:2600])
    data.append(temp[210:2600])


with open("30kg_1spring.csv","r") as file:
    reader = csv.reader(file, delimiter=',')
    row_num = 0
    for row in reader:
        if row_num == 2:
            temp = row
        row_num += 1
    temp = [float(i) for i in temp]
    #plt.plot(temp[1350:-1])
    data.append(temp[1350:-1])
    
with open("30kg_1spring_again.csv","r") as file:
    reader = csv.reader(file, delimiter=',')
    row_num = 0
    for row in reader:
        if row_num == 2:
            temp = row
        row_num += 1
    temp = [float(i) for i in temp]
    #plt.plot(temp[200:-1])
    data.append(temp[200:-1])
    
with open("10kg_3springs.csv","r") as file:
    reader = csv.reader(file, delimiter=',')
    row_num = 0
    for row in reader:
        if row_num == 2:
            temp = row
        row_num += 1
    temp = [float(i) for i in temp]
    plt.plot(temp[270:1200])
    data.append(temp[270:1200])
    
with open("20kg_3springs.csv","r") as file:
    reader = csv.reader(file, delimiter=',')
    row_num = 0
    for row in reader:
        if row_num == 2:
            temp = row
        row_num += 1
    temp = [float(i) for i in temp]
    #plt.plot(temp[750:3850])
    data.append(temp[0:680])
    data.append(temp[750:3850])
    
with open("30kg_3springs.csv","r") as file:
    reader = csv.reader(file, delimiter=',')
    row_num = 0
    for row in reader:
        if row_num == 2:
            temp = row
        row_num += 1
    temp = [float(i) for i in temp]
    #plt.plot(temp[1450:1700])
    data.append(temp[:240])
    data.append(temp[275:1430])
    data.append(temp[1450:1700])
    
    
    
    
with open("20kg_1spring_Elias_forced.csv","r") as file:
    reader = csv.reader(file, delimiter=',')
    row_num = 0
    for row in reader:
        if row_num == 2:
            temp = row
        row_num += 1
    temp = [float(i) for i in temp]
    #plt.plot(temp[2080:2800])
    data.append(temp[70:900])
    data.append(temp[900:1450])
    data.append(temp[1680:1850])
    data.append(temp[2080:2800])
    
with open("30kg_1spring_Elias.csv","r") as file:
    reader = csv.reader(file, delimiter=',')
    row_num = 0
    for row in reader:
        if row_num == 2:
            temp = row
        row_num += 1
    temp = [float(i) for i in temp]
    #plt.plot(temp[450:-1])
    data.append(temp[450:-1])

#TO BE COMPLETED FROM HERE
#with open("mass-spring-MAH01521.csv","r") as file:
#    reader = csv.reader(file, delimiter=',')
#    row_num = 0
#    for row in reader:
#        if row_num == 4:
#            temp = row
#        row_num += 1
#   temp = [float(i) for i in temp]
#    plt.plot(temp[:])

    
    
with open("data.csv","w") as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerows(data)
    
f1=DFT(data[0])
f2=DFT(data[1])
f3=DFT(data[2])
f4=DFT(data[3])
f5=DFT(data[4])
f6=DFT(data[5])
f7=DFT(data[6])
f8=DFT(data[7])
f9=DFT(data[8])
f10=DFT(data[9])
f11=DFT(data[10])
f12=DFT(data[11])
#f13=DFT(data[12]) deze is echt strange
f14=DFT(data[13])

m1=10
m2=20
m3=30

A1=np.array([[1/(m1*f1*f1)],[1/(m3*f2*f2)],[1/(m3*f3*f3)]])

A2=np.array([[1/(m1*f4*f4)],[1/(m2*f5*f5)],[1/(m2*f6*f6)],[1/(m2*f6*f6)],[1/(m3*f7*f7)],[1/(m3*f8*f8)],[1/(m3*f9*f9)]])
A3=np.array([[1/(m2*f10*f10)],[1/(m2*f11*f11)],[1/(m2*f12*f12)],[1/(m3*f14*f14)]]) #[1/(m2*f13*f13)]



C1=np.ones((len(A1),1))*4*np.pi*np.pi
C2=np.ones((len(A2),1))*4*np.pi*np.pi
C3=np.ones((len(A3),1))*4*np.pi*np.pi
k1=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A1),A1)),np.transpose(A1)),C1)
k13series=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A2),A2)),np.transpose(A2)),C2)
k1Elias=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A3),A3)),np.transpose(A3)),C3)

print(k1,k13series,k1Elias)


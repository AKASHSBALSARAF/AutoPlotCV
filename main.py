import numpy as np
import matplotlib.pyplot as plt

#first commit; nothing to see.

import cv2

#The values a,b,c,d are all placeholders for the corners of the actual image, starting from LB, LT, RT and RB in the order.

a=1
b=2
c=3
d=4

src = np.array([[0,0],[0,256],[256,256],[256,0]])
dst = np.array([[0,0],[0,256],[256,256],[256,0]])

x, y, u, v = src[:,0], src[:,1], dst[:,0], dst[:,1]
A = np.zeros((9,9))
j = 0
for i in range(4):
    A[j,:] = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*u[i], y[i]*u[i], u[i]])
    A[j+1,:] = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*v[i], y[i]*v[i], v[i]])
    j += 2
A[8, 8] = 1   # assuming h_9 = 1
b = [0]*8 + [1]

H = np.reshape(np.linalg.solve(A, b), (3,3))
print(H)

ps, p1s = [], [] 

for i in range(4):
    p, q = src[i], dst[i]
    p_hom = [p[0], p[1],1] 
    p1_hom = H@p_hom
    p1 = [x / p1_hom[2] for x in p1_hom[:-1]]
    ps.append(p)
    p1s.append(p1)
    print("P:{}, P':{}, H@P:{}".format(p, q, p1))


ps, p1s = np.array(ps), np.array(p1s)
max_y = 256
plt.figure(figsize=(10,3))
plt.subplot(121),   plt.scatter(ps[:,0], max_y-ps[:,1], color='g', s=50), plt.title('P', size=20), plt.grid()
plt.subplot(122),   plt.scatter(p1s[:,0], max_y-p1s[:,1], color='r', s=50), plt.title(r'P$^{\prime}$', size=20), plt.grid()
plt.show()
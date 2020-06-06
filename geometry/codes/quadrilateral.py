#code by Kamal Raj


import numpy as np
import matplotlib.pyplot as plt
from coeffs import *


#Variables required for making geometry
theta = 60*np.pi/180				#60 degree converted to radians
phi = 15*np.pi/180				#15 degree converted to radians
a = 3

#Finding Angles of trianlges
anglesum = 180*np.pi/180			#For finding third angle when two angles are known
alpha = (anglesum-theta)/2			#Two sides are equal and angle is known
beta = (anglesum-theta-phi-alpha)

#For finding all Sides
b = (a*np.sin(theta))/np.sin(alpha)		#Calculated using cosine formula
c = (a*np.sin(theta+phi))/np.sin(beta)
d = (a*np.sin(alpha))/np.sin(beta)
e = (d*np.sin(theta))/np.sin(alpha)

#Finding Vertices A(p,q)
p = b/2					#Using distance formula
q = np.sqrt((a**2)-(p**2))			#using distance formula

#Finding Vertices E(r,s)
thetaone = anglesum-(beta+alpha)		#angle requied to find sides

co = e*np.cos(thetaone)			#Distance between point C and O
r = c+co					#Distance between B and O also, it is equal to x co-ordinates of vertices E

height = co*np.tan(thetaone)			#It is equal to the y co-ordinated of vertices E
s = height


#Vertices array
A = np.array([p,q]) 
B = np.array([0,0])
C = np.array([c,0])
D = np.array([b,0])  
E = np.array([r,s])

#Print all vertices
print(A)
print(B)
print(C)
print(D)
print(E)

#Generating all lines
x_AB = line_gen(A,B)
x_BD = line_gen(B,D)
x_DC = line_gen(D,C)
x_CA = line_gen(C,A)
x_AD = line_gen(A,D)
x_CE = line_gen(C,E)
x_DE = line_gen(D,E)
x_EA = line_gen(E,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$')
plt.plot(x_DC[0,:],x_DC[1,:],label='$DC$')
plt.plot(x_CA[0,:],x_CA[1,:], linestyle='dashed', label='$CA$')
plt.plot(x_AD[0,:],x_AD[1,:], linestyle='dashed', label='$AD$')
plt.plot(x_CE[0,:],x_CE[1,:],label='$CE$')
plt.plot(x_DE[0,:],x_DE[1,:], linestyle='dashed', label='$DE$')
plt.plot(x_EA[0,:],x_EA[1,:],label='$EA$')


plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')
plt.plot(D[0], D[1], 'o')
plt.text(D[0] * (1 + 0.03), D[1] * (1 - 0.1) , 'D')
plt.plot(E[0], E[1], 'o')
plt.text(E[0] * (1 + 0.03), E[1] * (1 - 0.1) , 'E')

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.savefig('../figs/quadrilateral.pdf')  
plt.savefig('../figs/quadrilateral.eps')  
plt.show()

#This is an implementation of the new SAV scheme and I just found that vim
#is not case sensitive??

import taichi as ti
import taichi.math as tm
import numpy as np
from scipy.sparse import diags, lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)


delta_t = 0.1 #time step
eta = 1 # 0.1
M = 1
gamma = 0.05
epsilon = 1 # 0.04
Re = 1
rho = 1/Re
m = 8 #grid point
x_0, x_m = 0, 1
y_0, y_m = 0, 1
h = (x_m - x_0) / m

t_e = 5
T = 10
C0 = 0
#M = 1

x, delta_x = np.linspace(x_0, x_m, m+1, retstep = True)
y, delta_y = np.linspace(y_0, y_m, m+1, retstep = True)
X, Y = np.meshgrid(x[1:m], y[1:m], indexing = 'ij')

#process x and y to fit to representing 2D grid points
x_mod = np.zeros((m-1)**2)
y_mod = y[1:m]
for i in range(m-2):
    x_mod[i*m : i*m+m] = x[i+1]
    y_mod = np.vstack((y_mod, y[1:m]))
y_mod = y_mod.flatten()
#print(x_mod.shape, y_mod.shape)

'''
Initial values
'''
@ti.func
def phi_init(x, y):
#input x and y should be meshgrid X and Y
    #phi =\
    #0.24*np.cos(np.pi*x)*np.cos(2*np.pi*y)+0.4*np.cos(np.pi*x)\
    #*np.cos(3*np.pi*y)
    #phi = 0.05*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    #phi = np.tanh(1/(np.sqrt(2)*epsilon)*(0.25 - np.sqrt((x-0.5)**2 + \
    #(y-0.5)**2)))
    phi = 0.25 + 0.4*np.random.rand(len(x))
    #return phi.flatten()
    return phi

#initial condition for u
@ti.func
def u_init(x, y):
    u1 = -np.sin(np.pi*x)**2*(np.sin(2*np.pi*y))
    #u1 = np.zeros(len(x))
    #u1 = 10*x*y

    u2 = np.sin(np.pi*y)**2*(np.sin(2*np.pi*x))
    #u2 = np.zeros(len(y))
    #u2 = -10*x*y

    u = np.vstack((u1, u2))
    return u

'''
preparing random functions
'''

#function to calculate f_0
@ti.func
def f_0(phi, epsilon):
    return 1./(4*epsilon**2) * (1-phi**2)**2

#function to calculate f_0^prime
@ti.func
def f_0_prime(phi, epsilon):
    return -(1-phi**2)*phi/epsilon**2

#function to calculate energy E
@ti.func
def E(phi, m, h):
    #gradient of phi wrt x
    gradx_phi = gradient_mat(m, h,np.ones((m**2, 1)), 1, 'x').dot(phi)
    #gradient of phi wrt y
    grady_phi = gradient_mat(m,h,np.ones((m**2, 1)), 1, 'y').dot(phi)
    #square of norm of grad phi
    norm_grad_phi = (np.square(gradx_phi) + np.square(grady_phi))*1/2
    f0 = f_0(phi, epsilon)
    #integrand
    integrand = f0 + norm_grad_phi
    #perform integration
    w_B = 0.5*h*h #boundary values' weight
    w = h*h #weight of inner grids' value
    w_C = 0.25*h*h #corner values' weight
    integral = 0
    for i in range(m**2):
        if i == 0 or i == m-1 or i == m**2-m or i == m**2-1:
            integral += integrand[i] * w_C
        elif i < m or i > m**2-m or i % m == 0 or i % m == m-1:
            integral += integrand[i] * w_B
        else:
            integral += integrand[i] * w
    return integral

#function to calculate the initial value of mu
@ti.func
def mu_init(m, phi, epsilon):
    #print(poisson_des(m, h, -1).shape)
    #print(phi.shape)
    #print(poisson_des(m, h, -epsilon**2).shape, phi.shape)
    poisson_term = poisson_des(m, h, -epsilon**2).dot(phi)

    nonlinear_term = f_0_prime(phi, epsilon)
    return poisson_term + nonlinear_term


'''
functions constructing operator matrices
'''

#poisson space descretization (periodic BC)
@ti.func
def poisson_des_p(m, h, s):
    laplacian = poisson_des(m, h, s)
    for i in range(m):
        laplacian[i, m*(m-1)+i] = s/(h**2)
        laplacian[m*(m-1)+i, i] = s/(h**2)
    #connect the first and the last columns
    for i in range(m):
        laplacian[i*m, (i+1)*m-1] = s/(h**2)
        laplacian[(i+1)*m-1, i*m] = s/(h**2)
    return laplacian

#function to build gradient matrix and also deal with the convection term
@ti.func
def gradient_mat(m, h, u, s, x_or_y):
#u is the first operand of the convection term (u,grad(x)), u size m**2x1
#u is a vector for the inner product with the gradient matrix
#s is a scalar to be multiplied to the gradient matrix
#x_or_y indicate it's gradient wrt x or wrt y
    if (x_or_y == 'x'):
    #x-component of gradient (gradient along x)
        grad = lil_matrix((m**2, m**2))
        for i in range(m):
            for j in range(m-2):
                grad[i*m+j+1, i*m+j] = -s/(2*h) * u[i*m+j+1]
                grad[i*m+j+1, i*m+j+2] = s/(2*h) * u[i*m+j+1]
        for i in range(m):
            grad[i*m, i*m] = -3*s/(2*h) * u[i*m]
            grad[i*m, i*m+1] = 4*s/(2*h) * u[i*m]
            grad[i*m, i*m+2] = -s/(2*h) * u[i*m]
            grad[(i+1)*m-1, (i+1)*m-1] = 3*s/(2*h) * u[(i+1)*m-1]
            grad[(i+1)*m-1, (i+1)*m-2] = -4*s/(2*h) * u[(i+1)*m-1]
            grad[(i+1)*m-1, (i+1)*m-3] = s/(2*h) * u[(i+1)*m-1]
        return grad.tocsr()
    else:
    #y-component of gradient (gradient along y)
        grad = lil_matrix((m**2, m**2))
        #first and last row of grid points
        for i in range(m):
            #first row
            grad[i, i] = -3*s/(2*h) * u[i]
            grad[i, m+i] = 4*s/(2*h) * u[i]
            grad[i, 2*m+i] = -s/(2*h) * u[i]
            #last row
            grad[m*m-1-i, m*m-1-i] = 3*s/(2*h) * u[m*m-1-i]
            grad[m*m-1-i, m*m-1-i-m] = -4*s/(2*h) * u[m*m-1-i]
            grad[m*m-1-i, m*m-1-i-2*m] = s/(2*h) * u[m*m-1-i]
        #middle rows of grid points
        for i in range((m-2)*m):
            grad[i+m, i] = -s/(2*h) * u[i+m]
            grad[i+m, i+2*m] = s/(2*h) * u[i+m]
        return grad.tocsr()


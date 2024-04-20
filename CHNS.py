import numpy as np
#import pytorch as pt
from scipy.sparse import diags, lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


delta_t = 0.1 #time step
eta = 0.1
M = 0.1
gamma = 0.05
epsilon = 0.04
Re = 100
rho = 1/Re
m = 20 #grid point
x_0, x_m = 0, 1
y_0, y_m = 0, 1
h = (x_m - x_0) / (m-1)

t_e = 10
T = 5
C0 = 0
M = 1

x, delta_x = np.linspace(x_0, x_m, m, retstep = True)
y, delta_y = np.linspace(y_0, y_m, m, retstep = True)
X, Y = np.meshgrid(x, y, indexing = 'ij')

#process x and y to fit to representing 2D grid points
x_mod = np.zeros(m**2)
y_mod = y
for i in range(m-1):
	x_mod[i*m : i*m+m] = x[i]
	y_mod = np.vstack((y_mod, y)) 
y_mod = y_mod.flatten()
#print(x_mod.shape, y_mod.shape)

'''
#initial condition for phi
def phi_init(x, y):
	first_numerator = np.sqrt((x - 0.75*np.pi)**2 + (y-np.pi)**2) - np.pi/4
	first_term = np.tanh(first_numerator/(2*epsilon*np.sqrt(2)))
	second_numerator = np.sqrt((x - 1.25*np.pi)**2 + (y-np.pi)**2)-np.pi/4
	second_term = np.np.tanh(first_numerator/(2*epsilon*np.sqrt(2)))
	return first_term * second_term
#initial condition for u
def u_init(x, y):
	u = 0*x + 0*y
	return u
'''

#initial condition for phi
def phi_init(x, y):
#input x and y should be meshgrid X and Y
	phi =\
	0.24*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)+0.4*np.cos(np.pi*x)\
	*np.cos(3*np.pi*y)
	#return phi.flatten()
	return phi

#initial condition for u
def u_init(x, y):
	u1 = -np.sin(np.pi*x)**2*(np.sin(2*np.pi*y))
	#print((-np.sin(np.pi*x)**2).shape, (np.sin(2*np.pi*y)).shape)
	u2 = np.sin(np.pi*y)**2*(np.sin(2*np.pi*x))
	#print("u1 dim:", u1.shape)#TODO
	u = np.vstack((u1, u2))
	#print("u: shape",u.shape)#TODO
	return u

#function to calculate f_0
def f_0(phi):
	return 1./4 * (1-phi**2)**2

#function to calculate f_0^prime
def f_0_prime(phi):
	return (1-phi**2)*phi

#function to calculate energy E
def E(phi, f0, m, h):
	#gradient of phi wrt x
	gradx_phi = gradient_mat(m, h,np.ones((m**2, 1)), 0.5, 'x').dot(phi)
	#gradient of phi wrt y
	grady_phi = gradient_mat(m,h,np.ones((m**2, 1)), 0.5, 'y').dot(phi)
	#square of norm of grad phi
	norm_grad_phi = np.square(gradx_phi) + np.square(grady_phi)
	f0 = f_0(phi)
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
def mu_init(phi):
	#print(poisson_des(m, h, -1).shape)
	#print(phi.shape)
	poisson_term = poisson_des(m, h, -1).dot(phi)
	nonlinear_term = (1-phi**2)*phi
	return poisson_term + nonlinear_term

#function to calculate initial pressure term
def p_init(u, phi, mu, eta, rho, m, h):#TODO
	#LHS-grad p
	grad = poisson_des(m, h, 1)
	#RHS
	#terms involving u
	#print(u.shape) #TODO
	convection_u1 = gradient_mat(m, h, u[0,:], 1, 'x').dot(u[0,:]) + \
			gradient_mat(m, h, u[1,:], 1, 'y').dot(u[0,:])
	convection_u2 = gradient_mat(m, h, u[0,:], 1, 'x').dot(u[1,:]) + \
			gradient_mat(m, h, u[1,:], 1, 'y').dot(u[1,:])
	convection_u = np.vstack((convection_u1, convection_u2))
	div_convection = div_val(m, h, convection_u, -rho)

	poisson_u1 = poisson_des(m, h, 1).dot(u[0,:])
	poisson_u2 = poisson_des(m, h, 1).dot(u[1,:])
	poisson_u = np.vstack((poisson_u1, poisson_u2))
	div_poisson = div_val(m, h, poisson_u, eta)

	#terms involving phi and mu
	grad_phi_x = gradient_mat(m, h, np.ones((m**2, 1)), -1, 'x')
	grad_phi_y = gradient_mat(m, h, np.ones((m**2, 1)), -1, 'y')
	grad_mu_x = gradient_mat(m, h, grad_phi_x, 1, 'x')
	grad_mu_y = gradient_mat(m, h, grad_phi_y, 1, 'y')
	grad_mu_term = grad_mu_x + grad_mu_y
	poisson_mu = poisson_des(m, h, -1).dot(phi)

	RHS = poisson_mu + grad_mu_term + div_convection + div_poisson
	#solve for p_initial
	p = spsolve(grad, RHS)
	return p


#boundary condition for 1st&2nd time step
#backward Euler method
def b_euler(delta_t, y, f, n):
#f is a function; y'=f
	y_new = y + delta_t*f(n+1)
	return y_new

	
#poisson space descretization(dirichlet BC)
def poisson_des(m, h, s):
	laplacian = lil_matrix((m**2, m**2))
	laplacian.setdiag(-4*np.ones(m**2))
	for i in range(m**2):
		if i % m != 0:
			laplacian[i,i-1] = 1/(h**2) * s
		if (i+1) % m != 0:
			laplacian[i, i+1] = 1/(h**2) * s
	for i in range(m**2 - m):
		laplacian[i, i+m] = 1/(h**2) * s
		laplacian[i+m, i] = 1/(h**2) * s
	return laplacian.tocsr()


#gradient descretization
def gradient_des(x, m, h):
	del_x = zeros(m,1)
	for i in range(0, m):
		del_x[i] = {
			i == 0: (x[i]-x[i])/h,
			i == m-1: (x[i]-x[i-1])/h
		}.get(True, (x[i+1]-x[i-1])/(2*h))
	return del_x

#function to build gradient matrix and also deal with the convection term
def gradient_mat(m, h, u, s, x_or_y):
#u is the first operand of the convection term (u,grad(x)), u size m**2x1
#u is a vector for the inner product with the gradient matrix
#s is a scalar to be multiplied to the gradient matrix
#x_or_y indicate it's gradient wrt x or wrt y
	if (x_or_y == 'x'):
	#x-component of gradient (gradient along x)
		grad = lil_matrix((m**2, m**2))
		for i in range(m**2 - m):
			grad[i, i+m] = u[i]/(2*h) * s
			grad[i+m, i] = -u[i+m]/(2*h) * s
		for i in range(0, m):
			grad[i,i] = -u[i]/h * s
			grad[i, i+m] = u[i]/h * s
			grad[i+m**2-m, i+m**2-m] = u[i+m**2-m]/h * s
			grad[i+m**2-2*m, i+m**2-m] = -u[i+m**2-2*m]/h * s
		return grad.tocsr()
	else:
	#y-component of gradient (gradient along y)
		grad = lil_matrix((m**2, m**2))
		for i in range(m**2-1):
			grad[i, i+1] = u[i]/(2*h) * s
			grad[i+1, i] = -u[i+1]/(2*h) * s
		for i in range(m):
			grad[i*m, i*m] = -u[i*m]/h * s
			grad[i*m, i*m+1] = u[i*m]/h * s
			grad[(i+1)*m-1, (i+1)*m-1] = u[(i+1)*m-1]/h * s
			grad[(i+1)*m-1, (i+1)*m-2] = -u[(i+1)*m-1]/h * s
			if i != 0:
				grad[i*m, i*m-1] = 0
			if i != (m-1):
				grad[(i+1)*m-1, (i+1)*m] = 0
		return grad.tocsr()

#function to generate scalar multiply matrix
def scalar_mul(s, m):
#s is the scalar, m is the size
	scalar = lil_matrix((m**2, m**2))
	for i in range(0, m**2):
		scalar[i][i] = s
	return csr_matrix(scalar)

#function for calculating divergence
#u is a vector with 2 components (2D array)
def div_val(m, h, u, s):
	div_x = gradient_mat(m, h, np.ones((m**2, 1)), s, 'x')
	div_y = gradient_mat(m, h, np.ones((m**2, 1)), s, 'y')
	div = div_x.dot(u[0,:]) + div_y.dot(u[1,:])
	return div

#function to generate biharmonic discretization matrix
def biharmonic(m, h, s):
	poisson = poisson_des(m, h, s)
	biharmonic = csr_matrix(poisson.dot(poisson))
	return biharmonic

#first step: solve for u_hat with NS equation (helper function to
#solve_full_u_hat)
#dim specifies to solve for u1 or u2
def solve_u_hat(u_n, u_n_minus, p, phi, delta_t, rho, eta, m, dim):
	#building LHS
	constant_coef = scalar_mul(3, m)
	#convection term on the LHS
	convection_LHS = gradient_mat(m, h, u_n[0,:], delta_t, 'x') + \
					gradient_mat(m, h, u_n[1,:], delta_t, 'y')
	#Laplacian term on the LHS
	s = -delta_t / rho
	laplacian = poisson_des(m, h, s)
	
	LHS = constant_coef + convection_LHS + laplacian

	#building RHS
	ind = 0 if dim == 1 else 1 #index for column for u_n and u_n_minus
	linear_term = scalar_mul(4, m).dot(u_n[ind,:]) - u_n_minus[ind,:]
	convection_term = gradient_mat(m, h, u_n[0,:], -delta_t, 'x').dot(\
				u_n_minus[ind,:])\
				+ gradient_mat(m, h, u_n[1,:], -delta_t, 'y').dot(\
				u_n_minus[ind,:])
	convection_term *= delta_t
	#Laplacian term
	s2 = delta_t*eta/rho #scalar for poisson for u_n_minus
	laplacian_term = poisson_des(m, h, s2).dot(u_n_minus[ind,:])
	#gradient term for mu and p
	mu = helper_mu(m, h, phi, epsilon, C0)
	x_or_y = 'x' if dim == 1 else 'y'
	p_s = -2*delta_t/rho
	gradient_term = gradient_mat(m, h, np.ones((m**2,1)), p_s, x_or_y)\
			.dot(p) + gradient_mat(m, h, phi, p_s, x_or_y).dot(mu)

	RHS = linear_term + convection_term + laplacian_term + gradient_term

	#now deal with Dirichlet BC
	for i in range(m):
		for j in range(m):
			if i==0 or i == m-1 or j == 0:
				RHS[i*m+j] = 0
				for k in range(m**2):
					LHS[i*m+j, k] = 0
				LHS[i*m+j, i*m+j] = 1
			if j == m-1:
				RHS[i*m+j] = i*h*(1-i*h) if dim == 1 else 0
				for k in range(m**2):
					LHS[i*m+j, k] = 0
				LHS[i*m+j, i*m+j] = 1
	
	#finally, solve for u
	u_hat = spsolve(LHS, RHS)
	return u_hat

#function to get full u_hat (with two components)
def solve_full_u_hat(u_n, u_n_minus, p, phi, delta_t, rho, eta, m):
	u1 = solve_u_hat(u_n, u_n_minus, p, phi, delta_t, rho, eta, m, 1)
	u2 = solve_u_hat(u_n, u_n_minus, p, phi, delta_t, rho, eta, m, 2)
	u_hat = np.vstack((u1, u2))
	return u_hat

#function for solving pressure p using Helmholtz decomposition
#u is the u_hat^(n+1), p_n is the p in previous time step
def solve_p(m, h, u, delta_t, p_n):
	#RHS (divergence of u_n+1)
	RHS = div_val(m, h, u, -1)
	#LHS(laplacian of p_n+1 - p_n)
	poisson_p = poisson_des(m, h, delta_t/2)
	#solve for p_diff
	p_diff = spsolve(poisson_p, RHS)
	p = p_diff + p_n
	return p

#helper function to calculate phi_n+1/2
#this is using CH to approximate phi_mid, for more accuracy
#maybe need to use CHNS to approximate phi_mid
def helper_phi_mid(m, h, delta_t, phi_n):
	#LHS 
	#scalar term
	identity = scalar_mul(m, 1)
	#biharmonic term
	biharmonic_ = biharmonic(m, h, delta_t)
	LHS = biharmonic_ + identity
	#RHS
	f_prime = poisson_des(m, h, delta_t).dot(f_0_prime(phi_n))
	RHS = f_prime + phi_n
	#solve for phi_middle
	phi_mid = spsolve(LHS, RHS)
	return phi_mid

#helper function to calculate r
def helper_r(m, h, phi_n, C0):
	f_0_n = f_0(phi)
	r = np.sqrt(E(phi, f_0_n, m, h) + C0)
	return r

#helper function to solve b_n
def helper_b(m, h, phi_mid, C0):
	numerator = f_0_prime(phi_mid)
	f_0_mid = f_0(phi_mid)
	denominator = np.sqrt(E(phi_mid, f_0_mid, m, h)+C0)
	return numerator / denominator

#helper: solve Ax=(b)
def A_inverse_b(m, h, delta_t, M, epsilon, poisson_b):
	#LHS (A)
	scalar_ = scalar_mul(m, 3/(2*delta_t))
	biharmonic_ = biharmonic(m, h, M*epsilon**2)
	A = scalar_ + biharmonic_
	#solve for A_inverse_poisson_b
	result = spsolve(A, poisson_b)
	return result

#helper: calculate gamma = -<A^-1 Laplacian(b), b>
def gamma(m, h, delta_t, M, epsilon, b):
	poisson_b = poisson_des(m, h, 1).dot(b)
	A_term = A_inverse_b(m, h, delta_t, M, epsilon, poisson_b)
	return -M/2 * A_term * b

#helper: compute <b, phi_n+1>, it will also return A^-1*g
def b_phi_inner_product(m, h, delta_t, M, epsilon, u_hat, phi_n,\
						phi_n_minus, u_n_minus, r, r_minus, b):
	#RHS (g_term)
	div_u_hat = div_val(m, h, u_hat, -1/2)
	scalar_phi = (2/delta_t + div_u_hat) * phi_n
	# divergence phi term
	div_phi = div_val(m, h, phi_n, -1)
	u_div_phi = (u_hat + u_n_minus)/2 * div_phi
	# phi_n-1 term
	scalar_phi_minus = -phi_n_minus/(2*delta_t)
	# laplacian b term
	laplacian_b_coef = 4*M/3*r - 2*M/3*(b*phi_n) + M/6*(b*phi_n_minus)\
		- M/3*r_minus
	laplacian_b = poisson_des(m, h, laplacian_b_coef)

	g = scalar_phi + u_div_phi + scalar_phi_minus + laplacian_b
	A_inverse_g = A_inverse_b(m, h, delta_t, M, epsilon, g)

	RHS = A_inverse_g * b
	#LHS
	LHS = 1 + gamma(m, h, delta_t, M, epsilon, b)
	
	result = RHS/LHS
	return result, A_inverse_g

#helper: calculate mu for a certain time step
def helper_mu(m, h, phi_n, epsilon, C0):
	poisson_phi = poisson_des(m, h, -epsilon**2).dot(phi_n)
	r = helper_r(m, h, phi_n, C0)
	f_term = r * helper_b(m, h, phi_n, C0)
	return poisson_phi + f_term
	
#function to solve phi after solving u_hat_n+1
#This function is using u_hat as u_n+1 and not the real u_n+1 when
#solving for phi. TODO
def solve_phi(m, h, delta_t, M, epsilon, u_hat, u_n_minus, phi_n,\
			phi_n_minus, r, b):
	#RHS
	b_phi_ip, RHS_2 = b_phi_inner_product(m, h, delta_t, M, \
		epsilon, u_hat, phi_n, phi_n_minus, u_n_minus,r, b)

	first_term_coef = M/2 * b_phi_ip
	poisson_b = poisson_des(m, h, first_term_coef).dot(b)
	RHS_1 = A_inverse_b(m, h, delta_t, M, epsilon, poisson_b)
	RHS = RHS_1 + RHS_2
	return RHS #phi_n+1 equals RHS

#this function calculate the real u_n+1
def solve_real_u(m, h, u_hat, p_n, p_n_plus, delta_t):
	p_diff = p_n_plus - p_n
	grad_p_diff_x = gradient_mat(m, h, np.ones((m**2,1)), delta_t/2, 'x')
	grad_p_diff_y = gradient_mat(m, h, np.ones((m**2,1)), delta_t/2, 'y')
	u_real_x = u_hat[0,:] + grad_p_diff_x
	u_real_y = u_hat[1,:] + grad_p_diff_y
	u_real = np.vstack((u_real_x, u_real_y))
	return u_real

#The code below will officially begin iterating through time

#calculate u, phi, p for the first few time steps
#this delta_t is the delta_t for the first few steps using Euler method
def Euler_one_step(m, h, delta_t, M, u_n, p, phi, rho, eta, \
				epsilon, C0):
	# solve for u_hat
	u_n_minus = np.zeros((m**2, 2))
	u_n_ = u_n / 4
	u_hat_1 = solve_u_hat(u_n_, u_n_minus, p, phi, delta_t, rho,\
					eta, m, 1)
	u_hat_2 = solve_u_hat(u_n_, u_n_minus, p, phi, delta_t, rho, \
					eta, m, 2)
	u_hat = np.vstack((u_hat_1*3, u_hat_2*3))
	# solve for p
	p_n = p
	p = solve_p(m, h, u_hat, delta_t, p_n) #this is p_n+1
	# solve for real u_n+1
	u = solve_real_u(m, h, u_hat, p_n, p, delta_t) #this is u_n+1

	# solve for phi with u_hat
	# first, calculate phi_mid, b_n and r
	phi_mid = helper_phi_mid(m, h, delta_t, phi)
	b_n = helper_b(m, h, phi_mid, C0)
	r = helper_r(m, h, phi, C0)
	# then solve for phi_n+1
	phi_n = phi/4
	phi_n_minus = np.zeros((m**2, 1))
	phi_n_plus = solve_phi(m, h, delta_t, M, epsilon, u_hat, u_n, phi_n, \
				phi_n_minus, r, b_n) #this is phi_n+1
	
	return u, p, phi_n_plus
	
#calculate u, phi, p for the time step using BDF2
def BDF2_one_step(m, h, delta_t, M, u_n, u_n_minus, p, phi_n, phi_n_minus,\
			rho, eta, epsilon, C0):
	# solve for u_hat
	u_hat_1 = solve_u_hat(u_n, u_n_minus, p, phi_n, delta_t, rho,\
					eta, m, 1)
	u_hat_2 = solve_u_hat(u_n, u_n_minus, p, phi_n, delta_t, rho, \
					eta, m, 2)
	u_hat = np.vstack((u_hat_1, u_hat_2))

	# solve for p
	p_n = p
	p = solve_p(m, h, u_hat, delta_t, p_n) #this is p_n+1
	# solve for real u_n+1
	u = solve_real_u(m, h, u_hat, p_n, p, delta_t) #this is u_n+1

	# solve for phi with u_hat
	# first, calculate phi_mid, b_n and r
	phi_mid = helper_phi_mid(m, h, delta_t, phi_n)
	b_n = helper_b(m, h, phi_mid, C0)
	r = helper_r(m, h, phi_n, C0)
	# then solve for phi_n+1
	phi_n_plus = solve_phi(m, h, delta_t, M, epsilon, u_hat, u_n, phi_n, \
				phi_n_minus, r, b_n) #this is phi_n+1
	
	return u, p, phi_n_plus

#function to solve regular BDF2 time stepping
# T is the total amount of time steps
# t_e is the total amount of time steps for Euler method
def time_stepping(m, h, delta_t, t_e, T, x, y, eta, rho, epsilon, M, C0):
	#calculate the initial values
	u_init_ = u_init(x, y)
	phi_init_ = phi_init(x, y)
	mu_init_ = mu_init(phi_init_)
	p_init_ = p_init(u_init_, phi_init_, mu_init_, eta, rho, m, h)

	#start time stepping
	#first do Euler time stepping 10 times
	delta_t_euler = delta_t / t_e
	u_n_minus = u_init_
	u_n = u_init_
	p = p_init_
	phi_n_minus = phi_init_
	phi_n = phi_n

	for i in range(t_e):
		if i == 0:
			u_n, p, phi_n = Euler_one_step(m, h, delta_t_euler, M, \
					u_n_minus, p, phi_n_minus, rho, eta, epsilon, C0)
			continue
		u, p, phi = BDF2_one_step(m, h, delta_t_euler, M, \
				u_n, u_n_minus, p, phi_n, phi_n_minus, rho, eta, \
				epsilon, C0)
		#update n_minus and n variables
		u_n_minus = u_n
		phi_n_minus = phi_n
		u_n = u
		phi_n = phi
	
	for i in range(T - 1):
		if i == 0:
			u_n_minus = u_init_
			phi_n_minus = phi_init_
		u, p, phi = BDF2_one_step(m, h, delta_t, M, u_n, u_n_minus, p,\
				phi_n, phi_n_minus, rho, eta, epsilon, C0)
		#update n_minus and n variable
		u_n_minus = u_n
		phi_n_minus = phi_n
		u_n = u
		phi_n = phi

	return u_n, p, phi_n		

def plotting(m, h, delta_t, t_e, T, x, y, eta, rho, epsilon, M, C0):
	u, p, phi = time_stepping(m, h, delta_t, t_e, T, x, y, eta, \
						rho, epsilon, M, C0)
	u_norm = np.sqrt(u[:,0]**2, u[:,1]**2)
	#X, Y = np.meshgrid(x, y, indexing = 'ij')
	plt.contourf(X, Y, u_norm)
	plt.axis('scaled')
	plt.colorbar()
	plt.show()

plotting(m, h, delta_t, t_e, T, x_mod, y_mod, eta, rho, epsilon, M, C0)

import argparse #written module__interface
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.integrate
from numpy.polynomial.polynomial import Polynomial

class SnarkThat():
    """
    Brief conceptualization of zkSNARK functionality.
    """

    def __init__(self): 
        #initialise logic gates as foo

        foo1, foo2, foo3, foo4, foo5 = np.zeros((3,8)), np.zeros((3,8)),np.zeros((3,8)),np.zeros((3,8)),np.zeros((3,8))

        #using a.s * b.s = c.s; assign values to gates:

        foo1[0,1], foo1[1,1], foo1[2,4]= 1,1,1
        # print('\n foo1 \n',foo1) * foo1[2,4]
        foo2[0,4], foo2[1,1], foo2[2,5] = 1,1,1
        # print('\n foo2 \n',foo2)
        foo3[0,6], foo3[1,0], foo3[2,6]= 1,1,1
        # print('\n foo3 \n',foo3)
        foo4[0,5], foo4[1,6], foo4[2,7] = 1,1,1
        # print('\n foo4 \n',foo4)
        foo5[0,0] = -5
        foo5[0,7], foo5[1,0], foo5[2,3] = 1,1,1
        # print('\n foo5 \n',foo5)

        self.A = np.vstack((foo1[0,:], foo2[0,:], foo3[0,:], 
                   foo4[0,:], foo5[0,:]))

        self.B = np.vstack((foo1[1,:], foo2[1,:], foo3[1,:], 
                        foo4[1,:], foo5[1,:]))

        self.C = np.vstack((foo1[2,:], foo2[2,:], foo3[2,:], 
                        foo4[2,:], foo5[2,:]))
        
        # s = np.array(['_one', 'a','b', '_out', 
        #    '_gate1', '_gate2', '_gate3', '_gate4'])
        
        #s = self.arithmetic_circuit(alpha, beta)

        
    
        self.r1cs = np.array([self.A,self.B,self.C])
        # """
        # function outputs Rank-1 Constraint System logic gates as (3x5x8)Matrix
        # where Matrix[0], Matrix[1] and Matrix[2] correspond to A, B and C respectively.
        # """
        # self.r1cs = np.array([self.A,self.B,self.C])
        # return self.r1cs 


    def qeval(self, alpha, beta):
        """
        evaluates a**3 * 2b - 5; where (a,b) corresponds to (alpha,beta)

        Parameters
        ----------
        a : int
            user input
        b : int
            user input
        
        """
        return alpha**3 * 2*beta - 5

   #Flatten qeval
    def gate1(self, alpha):
        """function to square input. 
        alpha * alpha

        Parameters
        ----------
        alpha : int
            provided by user.
        """
        return alpha*alpha

    def gate2(self, alpha):
        """function multiplies input by gate1. 
        alpha**2 * alpha
        
        Parameters
        ----------
        alpha : int
            provided by user.
        """
        return self.gate1(alpha) * alpha

    def gate3(self, beta):
        """function multiplies input by 2. 
        beta * 2
        
        Parameters
        ----------
        beta : int
            provided by user.
        """
        return beta*2

    def gate4(self, alpha,beta):
        """function multiplies alpha**3 by beta*2. 
        (alpha**3) * (beta * 2)
        
        Parameters
        ----------
        alpha : int
            provided by user
        beta : int
            provided by user.
        """
        return self.gate2(alpha)*self.gate3(beta)

    def gate5(self, alpha,beta):
        """function subtracts 5 from (alpha**3) * (beta * 2).
        (alpha**3) * (beta * 2) - 5
        
        Parameters
        ----------
        alpha : int
            provided by user
        beta : int
            provided by user.
        """
        return self.gate4(alpha,beta) - 5

    def arithmetic_circuit(self, alpha, beta):
        """
        flattens function a**3 * 2b - 5

        Parameters
        ----------
        alpha : int
            provided by user
        beta : int
            provided by user.
        """
        _one = 1 
        _gate1 = self.gate1(alpha)
        _gate2 = self.gate2(alpha)
        _gate3 = self.gate3(beta)
        _gate4 = self.gate4(alpha,beta)
        _out = self.gate5(alpha,beta)

        s = np.array([_one, alpha, beta, _out, 
            _gate1, _gate2, _gate3, _gate4])
        
        return s

    #QAP:
    
    def poly(self, xi, yi):
        """
        function computes lagrange polynomial for set of points (x,y).

        Paramters
        ---------
        xi : array
            arbitrary x-coordinates
        yi : array
            y-coordinates corresponding to xi
        """
        return si.lagrange(xi,yi)

    def A_poly_coeffs(self):
        """
        compute coefficients of A polynomials 
        """
        xi = np.arange(5) #arbitrary 'x-co-ords' N+1 data points to N degree polynomial
        coeffs = []
        for i in range(8):
            yi = self.r1cs[0][:,i]
            coeffs.append(self.poly(xi, yi).coef[::-1])
        for i in range(8):
            #for loop generates 5x8 matrix
            if len(coeffs[i]) < 5:
                coeffs[i] = np.zeros(5)
        return np.stack(coeffs)
    
    def B_poly_coeffs(self):
        """
        compute coefficients of A polynomials 
        """
        xi = np.arange(5) #arbitrary 'x-co-ords' N+1 data points to N degree polynomial
        coeffs = []
        for i in range(8):
            yi = self.r1cs[1][:,i]
            coeffs.append(self.poly(xi, yi).coef[::-1])
        for i in range(8):
            #for loop generates 5x8 matrix
            if len(coeffs[i]) < 5:
                coeffs[i] = np.zeros(5)
        return np.stack(coeffs)
    
    def C_poly_coeffs(self):
        """
        compute coefficients of A polynomials 
        """
        xi = np.arange(5) #arbitrary 'x-co-ords' N+1 data points to N degree polynomial
        coeffs = []
        for i in range(8):
            yi = self.r1cs[2][:,i]
            coeffs.append(self.poly(xi, yi).coef[::-1])
        for i in range(8):
            #for loop generates 5x8 matrix
            if len(coeffs[i]) < 5:
                coeffs[i] = np.zeros(5)
        return np.stack(coeffs)


    #QAP proof:
    # s.A(x) * s.B(x) = s.C(x)
    
    def all_poly_coeffs(self):
        """
        returns A, B, C polynomials coefficient
        """
        
        print('\n polynomial A:\n', self.A_poly_coeffs(),'\n \n polynomial B:\n', self.B_poly_coeffs(),'\n \n polynomial C:\n', self.C_poly_coeffs())

    def plot(self,x,y,ax,i,title):
    
        ax.plot(x,y,label = 'QAP function:{0:1d}'.format(i))
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('$x$', fontsize=12)
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)
        
    
    def plot_poly(self):
    
        x = np.linspace(0, 5, 100)
        f1 = self.A_poly_coeffs()
        f2 = self.B_poly_coeffs()
        f3 = self.C_poly_coeffs()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

        for i in range(8):
            y = f1[i,4]*x**4 + f1[i,3]*x**3 + f1[i,2]*x**2 + f1[i,1]*x + f1[i,0]
            self.plot(x,y,ax1,i, 'A polynomials')

        for i in range(8):
            y = f2[i,4]*x**4 + f2[i,3]*x**3 + f2[i,2]*x**2 + f2[i,1]*x + f2[i,0]
            self.plot(x,y,ax2,i, 'B polynomials')

        for i in range(8):
            y = f3[i,4]*x**4 + f3[i,3]*x**3 + f3[i,2]*x**2 + f3[i,1]*x + f3[i,0]
            self.plot(x,y,ax3,i, 'C polynomials') 
        plt.show()
        
    def verify(self, alpha, beta):
        """
        s.A(x) * s.B(x) - s.C(x) = 0
        evaluating at x = 1
        """
        s = self.arithmetic_circuit(alpha,beta)
        
        mat_A = self.A_poly_coeffs()
        mat_B = self.B_poly_coeffs()
        mat_C = self.C_poly_coeffs()
        
        s_A = np.dot(s, mat_A)
        s_B = np.dot(s, mat_B)
        s_C = np.dot(s, mat_C)
            
        if alpha<abs(10) | beta<abs(10):
            #control measure for overflow due exponential large numbers.
            lhs = np.exp(s_A.sum()*s_B.sum())
            rhs = np.exp(s_C.sum())
        else:
            lhs = s_A.sum()*s_B.sum()
            rhs = s_C.sum()

        acc_metric1 = abs(rhs - lhs)/lhs * 100 #a measure of proximity between s.A*s.B and s.C
        
        print('\n verify: ',np.isclose(lhs,rhs), '\n proximity%: ',acc_metric1)
        

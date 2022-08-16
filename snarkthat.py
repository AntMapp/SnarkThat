import argparse #written module__interface
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.integrate
from numpy.polynomial.polynomial import Polynomial

class SnarkThat():
    """
    
    """

    def __init__(self): 
        #initialise logic gates as foo

        foo1, foo2, foo3, foo4, foo5 = np.zeros((3,8)), np.zeros((3,8)),np.zeros((3,8)),np.zeros((3,8)),np.zeros((3,8))

        #using a.s * b.s = c.s; assign values to gates:

        foo1[0,1], foo1[1,1], foo1[2,3]= 1,1,1
        # print('\n foo1 \n',foo1)
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
            if len(coeffs[i]) < 5:
                coeffs[i] = np.zeros(5)
        return np.stack(coeffs)


    #QAP proof:
    # s.A(x) * s.B(x) = s.C(x)

    def verify(self, alpha, beta):
        """
        s.A(x) * s.B(x) - s.C(x) = 0
        evaluating at x = 1
        """
        s = self.arithmetic_circuit(alpha,beta)
        matrixA = self.A_poly_coeffs()
        matrixB = self.B_poly_coeffs()
        matrixC = self.C_poly_coeffs()

        return np.dot(s, matrixA[:,0].T) * np.dot(s, matrixB[:,0].T) - np.dot(s, matrixC[:,0].T)
    
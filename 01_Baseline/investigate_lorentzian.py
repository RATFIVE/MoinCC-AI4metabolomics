import numpy as np
import matplotlib.pyplot as plt

def lorentzian(x, shift, gamma, A):
    '''
    x is the datapoint
    x0 is the peak position
    gamma is the width
    A is the amplitude
    '''
    return A * gamma / ((x - shift)**2 + gamma**2)

shift = 0
gamma = 1e-1
A = 300
x = np.linspace(-10, 10, 1000)
y = lorentzian(x, shift=shift, gamma=gamma, A=A)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lorentzian')
plt.legend([f'x0={shift}, gamma={gamma}, A={A}'])   
plt.show()

# -*- coding: utf-8 -*-
"""
November 5th, 2020

@author: Christopher Aceto

This is a simulation of the analog feedback system for
magnetic levitation described in "A levitated magnetic
dipole configuration as a compact charged particle trap"
(Rev. Sci. Instrum. 91, 043507, 2020),
but with updated parameters and a more accurate model
than is practical for analytical methods.
"""

from tqdm import trange #shows a progress bar for simulation
from random import random #random number generator for white noise
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import winsound #to play a sound when simulation is finished


"""
System Parameters
"""
R63 = 35000. #differentiator's variable resistor, 100 to 200100 ohms
p = 8. #proportional gain
zeq = 0.226 #equilibrium F coil position, meters below L coil
zoff = 0.061 #offset between laser rangefinder and center of L coil, minus half of height of F coil
#(measurement of laser rangefinder is different from actual F coil position below L coil)
amplitude = 0. #noise amplitude, volts
ilo = 39. #equilibrium L coil current (amps) depends on F coil current; can be calculated

#initial values: [z, dz/dt]
init = np.array([[.24, 0.]])

#seconds, tfinal/tstep should be integer
tfinal = .05
tstep = .000005
textra = .03 #for smoothing out convolutions. Should be at least .02; .03 is better
extraNum = round(textra / tstep) #extra number of time steps due to textra

#CAUTION: Don't set any of w2,w3,w5 the same as another
w2 = 500.        #1/s
w3 = 314.        #1/s
w5 = 4550.       #1/s
gamma2 = 25.     #A/V
gamma5 = 100./3. #V/m
g = 9.8          #m/s^2


"""
Function describing decay (or variation) of I_F.
decay(t) is unitless. I_F = I_F0 * decay(t)
"""
tau = 30 #seconds, time constant of decay
freq = 2 #Hz, frequency of oscillation
Famp = .05 #oscillation amplitude
def decay(t):
    return 1
    #np.exp(-t/tau)
    #(1 + Famp * np.sin(2*np.pi*freq*t))

"""
h/heq as function of z, using zeq == .226 to calculate heq.
This is a good rational approximation; under 0.1% error in operating range.
h is radial component of magnetic field experienced by F coil, divided by I_L.
"""
def h(z):
    return 0.277 + (0.72138*z - 0.11037) / (z*z - 0.08349*z + 0.04063)
    
"""
Function for the reference voltage, which creates a setpoint for the system.
Required vref for a desired zeq depends on various parameters.
vref can be varied to adjust F coil equilibrium position. Changing vref linearly is commented out below.
"""
#timeStopChngVref = 1.
#slope = (zeq - init[0,0])/timeStopChngVref
#vref0 = gamma5*(.236+zoff) - ilo/(gamma2*p)
vref_ = gamma5*(zeq+zoff) - ilo/(gamma2*p)
def vref(t):
    return vref_
    '''
    if t < timeStopChngVref:
        zeq_t = slope*t + init[0,0]
        return gamma5*zeq_t - (ilo/(gamma2*p)) / decay(t)
    else:
        return gamma5*zeq - (ilo/(gamma2*p)) / decay(t)'''

"""
Time values.
"""
t = np.linspace(0.0, tfinal, num=(round((tfinal/tstep)) + 1))

"""
Electric noise. Currently, inserted in input of PID circuit.
"""
noise = amplitude * np.sin(2 * np.pi * 50 * t) #50 Hz line noise
#amplitude * np.exp(-(t - 0.07)**2 / (2 * .001**2)) #Gaussian function
'''
noise = np.empty(np.size(t)) #random (white) noise
for i in range(np.size(noise)):
    noise[i] = -amplitude + (random() * 2*amplitude)'''



"""
Combinations of constants to reduce computation time.
"""
c1 = gamma2 * gamma5 * w2 * w3 * w5 / ilo
c2 = gamma2 / ilo
cD1 = 42727.3 / (.00094 - 0.000000022 * R63)
cD2 = R63 / (.00094 - 0.000000022 * R63)
cD3 = 0.0000000022 * R63

"""
Functions of t to reduce computation time.
3 sets for 3 different t inputs in RK4 algorithm.
"""
#convolution of time-domain versions of transfer functions G2, G3, G5
func1a = c1 * ((np.exp(t * -w2) - np.exp(t * -w5))
                  / ((w2 - w3) * (w2 - w5))
              + (np.exp(t * -w3) - np.exp(t * -w5))
                  / ((w3 - w2) * (w3 - w5)))
#time-domain version of differentiator transfer function
func2a = cD1 * np.exp(t / -cD3) \
             - cD2 * np.exp(t / -.000094)


func1b = c1 * ((np.exp((t + 0.5*tstep) * -w2)
                      - np.exp((t + 0.5*tstep) * -w5))
                  / ((w2 - w3) * (w2 - w5))
              + (np.exp((t + 0.5*tstep) * -w3)
                      - np.exp((t + 0.5*tstep) * -w5))
                  / ((w3 - w2) * (w3 - w5)))
func2b = cD1 * np.exp((t + 0.5*tstep) / -cD3) \
             - cD2 * np.exp((t + 0.5*tstep) / -.000094)


func1c = c1 * ((np.exp((t + tstep) * -w2) - np.exp((t + tstep) * -w5))
                  / ((w2 - w3) * (w2 - w5))
              + (np.exp((t + tstep) * -w3) - np.exp((t + tstep) * -w5))
                  / ((w3 - w2) * (w3 - w5)))
func2c = cD1 * np.exp((t + tstep) / -cD3) \
             - cD2 * np.exp((t + tstep) / -.000094)

#add extra values to smooth out convolutions
func1a = np.concatenate((func1a, func1a[-1] * np.ones(extraNum)))
func1b = np.concatenate((func1b, func1b[-1] * np.ones(extraNum)))
func1c = np.concatenate((func1c, func1c[-1] * np.ones(extraNum)))
func2a = np.concatenate((func2a, func2a[-1] * np.ones(extraNum)))
func2b = np.concatenate((func2b, func2b[-1] * np.ones(extraNum)))
func2c = np.concatenate((func2c, func2c[-1] * np.ones(extraNum)))


"""
Derivative function. q is solution array ([[z0,v0],[z1,v1],...]).
Returns derivatives ([dz/dt, (d^2 z)/(dt^2)])
3 versions for 3 sets of functions above (to save time)
"""
def D1(q, t):
    dv = g - g * decay(t[i]) * \
            (tstep * convolve(
                func1a[:(i+1 + extraNum)],
                p * (q[:(i+1 + extraNum)][:,0] + zoff)
                + tstep * convolve(
                    (q[:(i+1 + extraNum)][:,0] + zoff),
                    func2a[:(i+1 + extraNum)])[:(i+1 + extraNum)], mode='valid')[0]
            + (noise[i] - vref(t[i])) * c2 * p) \
            * h(q[i + extraNum][0])
    return np.array([q[i + extraNum][1], dv])

def D2(q, t):
    dv = g - g * decay(t[i]) * \
            (tstep * convolve(
                func1b[:(i+1 + extraNum)],
                p * (q[:(i+1 + extraNum)][:,0] + zoff)
                + tstep * convolve(
                    (q[:(i+1 + extraNum)][:,0] + zoff),
                    func2b[:(i+1 + extraNum)])[:(i+1 + extraNum)], mode='valid')[0]
            + (noise[i] - vref(t[i])) * c2 * p) \
            * h(q[i + extraNum][0])
    return np.array([q[i + extraNum][1], dv])

def D3(q, t):
    dv = g - g * decay(t[i]) * \
            (tstep * convolve(
                func1c[:(i+1 + extraNum)],
                p * (q[:(i+1 + extraNum)][:,0] + zoff)
                + tstep * convolve(
                    (q[:(i+1 + extraNum)][:,0] + zoff),
                    func2c[:(i+1 + extraNum)])[:(i+1 + extraNum)], mode='valid')[0]
            + (noise[i] - vref(t[i])) * c2 * p) \
            * h(q[i + extraNum][0])
    return np.array([q[i + extraNum][1], dv])



"""
Numerical solution to differential equation.
Solution array column 0 is positions, column 1 is velocities.
"""
solution = np.concatenate((init, np.empty((np.size(t) - 1,2))))
#add extra values for smoothing out convolutions
solution = np.concatenate((solution[0] * np.ones((extraNum,2)), solution))

#RK4 loop
for i in trange(np.size(t) - 1, position=0, leave=True):
    k1 = tstep * D1(solution, t)
    k2 = tstep * D2(solution + 0.5*k1, t + 0.5*tstep)
    k3 = tstep * D2(solution + 0.5*k2, t + 0.5*tstep)
    k4 = tstep * D3(solution + k3, t + tstep)
    
    solution[i+1 + extraNum] = \
        solution[i + extraNum] + (k1 + 2*k2 + 2*k3 + k4)/6



"""
Visualization. Plots position in blue, velocity in orange.
Position (of center of F coil) is measured as meters below center of L coil.
Due to this, positive velocity means downward motion.
"""
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('seconds; tstep='+str(tstep), fontsize=12)
ax2 = ax.twinx()

#plot position and velocity
ax.plot(t, solution[extraNum:][:,0])
ax2.plot(t, solution[extraNum:][:,1], 'orange')
ax.set_ylabel('z (m below L coil)', color='C0', fontsize=18)
ax2.set_ylabel('dz/dt (m/s)', color='orange', fontsize=18)

fig.suptitle('z0='+str(init[0][0])+' v0='+str(init[0][1])+' R63='+str(R63)+
             ' P='+str(p)+' ilo='+str(ilo))
ax.set_title('g2='+str(gamma2)+' w2='+str(w2)+' w3='+str(w3)+
             ' g5='+str(round(gamma5,3))+' w5='+str(w5))
plt.show()


#sound to signal completion
winsound.Beep(440,200)
winsound.Beep(528,200)
winsound.Beep(660,200)
winsound.Beep(792,200)
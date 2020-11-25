# -*- coding: utf-8 -*-
"""
November 5th, 2020

@author: Christopher Aceto

This program uses a binary search algorithm to find the
highest P, for a given resistance of R63 in the
differentiator, that creates a non-oscillatory response
to displacement without overshoot.

Program outputs a list:
[R63 value used, 'best' P found,
    time system took to return to within 1% of equilibrium,
    initial position, initial velocity]

A binary search can be used because the system is always
more stable for a lower P (and less stable for a higher P),
except for P below about 0.9.

The velocity is tested for monotonicity because this is
more strict than testing the position.
"""
from tqdm import trange #shows a progress bar for simulation
import numpy as np
from scipy.signal import convolve
import winsound #to play a sound when simulation is finished


"""
System Parameters
"""
R63 = 20000. #differentiator's variable resistor, 100 to 200100 ohms
plow = .8 #initial lower P bound for binary search
phigh = 1.2 #initial upper P bound for binary search
pPrecision = 1   #no. of decimal places for P, integer

zeq = 0.226 #equilibrium F coil position, meters below L coil
zoff = 0.061 #offset between laser rangefinder and center of L coil, minus half of height of F coil
#(measurement of laser rangefinder is different from actual F coil position below L coil)
ilo = 39. #equilibrium L coil current (amps) depends on F coil current; can be calculated

#initial values: [z, dz/dt], where z is distance below center of L coil, meters
init = np.array([[.24, 0.]])

#seconds, tfinal/tstep should be integer
tfinal = .5
tstep = .000005
textra = .03 #for smoothing out convolutions. Should be at least .02; .03 is better
extraNum = round(textra / tstep) #extra number of time steps due to textra

tcheck = .13 #time to start checking velocity for monotonicity
#use the other program to determine when the velocity should begin to be monotonic

#2 is L coil current source, 3 is eddy currents in vacuum chamber walls, 5 is laser rangefinders
#CAUTION: Don't set any of w2,w3,w5 the same as another
w2 = 250.        #1/s
w3 = 500.        #1/s
w5 = 4550.       #1/s
gamma2 = 25.     #A/V
gamma5 = 100./3. #V/m
g = 9.8          #m/s^2


"""
Function describing decay (or variation) of I_F.
decay(t) is unitless. I_F = I_F0 * decay(t)
"""
tau = 3600 #seconds, time constant of decay
def decay(t):
    return 1 #np.exp(-t/tau)

"""
h/heq as function of z, using zeq == .226 to calculate heq.
This is a good rational approximation; under 0.1% error in operating range.
h is radial component of magnetic field experienced by F coil, divided by I_L.
"""
def h(z):
    return 0.277 + (0.72138*z - 0.11037) / (z*z - 0.08349*z + 0.04063)

"""
Time values.
"""
t = np.linspace(0.0, tfinal, num=(round((tfinal/tstep)) + 1))



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
def D1(q, t, p, vref, i):
    dv = g - g * decay(t[i]) * \
            (tstep * convolve(
                func1a[:(i+1 + extraNum)],
                p * (q[:(i+1 + extraNum)][:,0] + zoff)
                + tstep * convolve(
                    (q[:(i+1 + extraNum)][:,0] + zoff),
                    func2a[:(i+1 + extraNum)])[:(i+1 + extraNum)], mode='valid')[0]
            - vref * c2 * p) \
            * h(q[i + extraNum][0])
    return np.array([q[i + extraNum][1], dv])

def D2(q, t, p, vref, i):
    dv = g - g * decay(t[i]) * \
            (tstep * convolve(
                func1b[:(i+1 + extraNum)],
                p * (q[:(i+1 + extraNum)][:,0] + zoff)
                + tstep * convolve(
                    (q[:(i+1 + extraNum)][:,0] + zoff),
                    func2b[:(i+1 + extraNum)])[:(i+1 + extraNum)], mode='valid')[0]
            - vref * c2 * p) \
            * h(q[i + extraNum][0])
    return np.array([q[i + extraNum][1], dv])

def D3(q, t, p, vref, i):
    dv = g - g * decay(t[i]) * \
            (tstep * convolve(
                func1c[:(i+1 + extraNum)],
                p * (q[:(i+1 + extraNum)][:,0] + zoff)
                + tstep * convolve(
                    (q[:(i+1 + extraNum)][:,0] + zoff),
                    func2c[:(i+1 + extraNum)])[:(i+1 + extraNum)], mode='valid')[0]
            - vref * c2 * p) \
            * h(q[i + extraNum][0])
    return np.array([q[i + extraNum][1], dv])



#tests whether an array is monotonic, returns boolean
def monotonic(x):
    #any numeric solution will oscillate. must round off some digits to actually be monotonic
    dx = np.diff(np.around(x,7))
    return np.all(dx <= 0) or np.all(dx >= 0)


"""
Finds numerical solution to differential equation.
Returns a solution array: one element for each time value,
and each element is a 2-element list of [position,velocity]
"""
def solve(p):
    #PID reference voltage, caluclated based on steady-state conditions
    vref = gamma5*(zeq+zoff) - ilo/(gamma2*p)
    
    #solution array
    solution = np.concatenate((init, np.empty((np.size(t) - 1,2))))
    #extra values for smoothing out convolutions
    solution = np.concatenate((solution[0] * np.ones((extraNum,2)), solution))
    
    #RK4 loop
    for i in trange(np.size(t) - 1, position=0, leave=True):
        k1 = tstep * D1(solution, t, p, vref, i)
        k2 = tstep * D2(solution + 0.5*k1, t + 0.5*tstep, p, vref, i)
        k3 = tstep * D2(solution + 0.5*k2, t + 0.5*tstep, p, vref, i)
        k4 = tstep * D3(solution + k3, t + tstep, p, vref, i)
        
        solution[i+1 + extraNum] = \
            solution[i + extraNum] + (k1 + 2*k2 + 2*k3 + k4)/6
    
    return solution



"""
Binary search for the best P value. Tests midpoint P; if that
gives monotonic velocity, then midpoint becomes new lower bound
and the midpoint is tested (or vice versa).

Stops once desired precision is reached.
"""
pmid = round((phigh+plow)/2, pPrecision)
while (pmid != phigh) and (pmid != plow):
    sol = solve(pmid)
    mono = monotonic(sol[round(tcheck/tstep) + extraNum:][:,1])
    if mono == True:
        plow = pmid
    else:
        phigh = pmid
    pmid = round((phigh+plow)/2, pPrecision)
    

"""
Another binary search to determine how long system takes to
return to within 1% of equilibrium, using previously found 'best' P.

This test assumes F coil started with some displacement.
"""
#indices of the solution array to serve as initial bounds
index1 = round(tcheck/tstep)
index2 = np.size(sol[extraNum:][:,0]) - 1
#midpoint
indexMid = round((index1 + index2)/2)
while (indexMid != index1) and (indexMid != index2):
    value = sol[indexMid + extraNum,0]
    if (np.abs(value - zeq) < (.01 * zeq)):
        index2 = indexMid
    else:
        index1 = indexMid
    indexMid = round((index1 + index2)/2)

timeTo1 = indexMid * tstep


print('[R63, P_best, time to 1%, z0, dz/dt0]')
print([R63,pmid,timeTo1,init[0][0],init[0][1]])


#sound to signal completion
winsound.Beep(440,200)
winsound.Beep(520,200)
winsound.Beep(660,200)
winsound.Beep(777,200)

#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Use Times type 1 fonts
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = '\usepackage{txfonts}'
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern']})

#===============================================================================
# NUMBER OF SCORING EVENTS

n_rings = range(1,11)
events = [21.6401, 43.8720, 65.4837, 86.5676, 107.958,
          128.884, 151.271, 170.658, 192.050, 213.058]

plt.plot(n_rings, events, 'o')
plt.xlim([0,10])
plt.xlabel('Number of annular regions', fontsize=24)
plt.ylabel('Tracks in fuel per particle, $f$', fontsize=24)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.grid(True, which='both')
plt.savefig('events.pdf', bbox_inches='tight')
plt.close()

#===============================================================================
# TRACKING TIME

factor = 412.113/np.array([412.113, 390.093, 385.791, 380.626, 369.459,
                           360.847, 351.098, 339.704, 337.623, 333.316])

plt.plot(n_rings, factor, 'o')
# plt.xlim([0,10])
plt.xlabel('Number of annular regions', fontsize=24)
plt.ylabel('Relative tracking time, $\mu/\mu_1$', fontsize=24)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.grid(True, which='both')
plt.savefig('time.pdf', bbox_inches='tight')
plt.close()

#===============================================================================
# TALLY SERVER MODEL

# Change default colors
matplotlib.rcParams['axes.color_cycle'] = [cm.jet(k) for k in np.linspace(0, 1, 10)]

# Parameters
mu    = 1./140.  # Particles per second on Titan Cray XK7
alpha = 2.0e-6   # Average ping-pong latency on Titan Cray XK7
beta  = 2.5e-10  # Bandwidth on Titan Cray XK7

# Parameters from OpenMC simulation
dmax = 15360
d = np.arange(8, dmax, 4) # Data/event (19.2 kb for depletion)

for i in range(10):
    # Calculate time for given numbers of events
    overhead = events[i]/(mu*factor[i])*(alpha + d*beta)

    # Plot overhead
    plt.loglog(d, overhead, label='{0} region{1}'.format(
            i+1, 's' if i else ''))

# Set plotting options
plt.xlim([0,dmax])
plt.xlabel(r'Data per event, $d$ (bytes)', fontsize=24)
plt.ylabel(r'Overhead per batch, $\Delta$', fontsize=24)
plt.gca().tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, which='both')
plt.legend(loc='upper left', ncol=2, prop={'size':16})
plt.savefig('model.pdf')
plt.close()

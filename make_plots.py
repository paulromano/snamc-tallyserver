#!/usr/bin/env python

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Use Times type 1 fonts
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = '\usepackage{txfonts}'
mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})

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
mpl.rcParams['axes.color_cycle'] = [cm.jet(k) for k in np.linspace(0, 1, 10)]

# Parameters
mu    = 1./69.   # Particles per second on Mira Blue Gene/Q
alpha = 2.5e-6   # Average latency on Mira Blue Gene/Q
beta  = 5.55e-10 # Bandwidth on Mira Blue Gene/Q

# Parameters from OpenMC simulation
dmax = 15360
d = np.arange(8, dmax, 4) # Data/event (15.36 kb for depletion)

for i in range(10):
    # Calculate time for given numbers of events
    overhead = events[i]/(mu*factor[i])*(alpha + d*beta)

    # Plot overhead
    plt.loglog(d, overhead, label='{0} region{1}'.format(
            i+1, 's' if i else ''))

# Set plotting options
plt.xlim([0,dmax])
plt.xlabel(r'Data per event, $d$ (bytes)', fontsize=24)
plt.ylabel(r'Overhead per batch, $\Delta_s$', fontsize=24)
plt.gca().tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, which='both')
plt.legend(loc='upper left', ncol=2, prop={'size':16})
plt.savefig('model.pdf')
plt.close()

#===============================================================================
# NEGATIVE OVERHEAD MODEL

mpl.rcParams['axes.color_cycle'] = mpl.rcParamsDefault['axes.color_cycle']

f = 21.3
mu_b = 5.0e-8

# Calculate time for given numbers of events
oldModel = f/mu*(alpha + d*beta)
newModel = (mu + f*(alpha + d*beta))/(mu + mu_b*d) - 1.

# Plot overhead
plt.semilogx(d, oldModel, 'k-', label='Eq. (5)'.format(f))
plt.semilogx(d, newModel, 'k--', label='Eq. (9)'.format(f))

# Set plotting options
plt.xlim([0,dmax])
plt.xlabel(r'Data per event, $d$ (bytes)', fontsize=24)
plt.ylabel(r'Overhead per batch, $\Delta_s$', fontsize=24)
plt.gca().tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, which='both')
plt.legend(loc='upper left', prop={'size':12})
plt.savefig('model_negative.pdf')
plt.close()

#===============================================================================
# RESULTS

procs = [16, 32, 64, 128, 256, 512]
colors = ['r','g','b','c','m','k']
ratio = [1, 3, 7, 15]
data = [240*2**i for i in range(7)]
overhead = {}

# Read baseline results
nucs, baseline0, baseline1 = np.loadtxt('data/baseline.txt', usecols=(0,2,3), unpack=True)
overheadBase = baseline0/baseline1

# Baseline plot
baselineMu = 1/baseline1
plt.plot(nucs*6*8, baselineMu/baselineMu[0], 'ko-')
plt.xlabel('d (bytes)', fontsize=24)
plt.ylabel('$\mu_t/\mu_{t,0}$', fontsize=24)
plt.gca().tick_params(axis='both', which='major', labelsize=16)
plt.ylim((0.8,2.0))
plt.grid(which='both')
plt.savefig('baseline.pdf')
plt.close()

# Read Mira BG/Q blocking comm results
runs = np.genfromtxt('data/block.txt', usecols=0, dtype=str)
inactiveB, activeB = np.loadtxt('data/block.txt', usecols=(2,3), unpack=True)

# Read Mira BG/Q non-blocking comm results
inactiveNB, activeNB = np.loadtxt('data/nonblock1.txt', usecols=(2,3), unpack=True)

overheadB = {}
overheadNB = {}
for i in range(len(inactiveB)):
    p, r, f, n = map(int, re.match(r'p(\d+)-s(\d+)-f(\d+)-n(\d+)',
                                   runs[i]).groups())
    d = n*6*8
    overheadB[p,r,d] = (1/activeB[i] - overheadBase[i%7]/inactiveB[i])*inactiveB[i]
    overheadNB[p,r,d] = (1/activeNB[i] - overheadBase[i%7]/inactiveNB[i])*inactiveNB[i]

# Plot Intrepid results
for r in ratio:
    for i, p in enumerate(procs):
        plt.semilogx(data, [overheadB[p,r,d] for d in data], colors[i] + '-',
                   label='$p={0}$, Blocking'.format(p))
        plt.semilogx(data, [overheadNB[p,r,d] for d in data], colors[i] + '--',
                   label='$p={0}$, Non-blocking'.format(p))

    # Set plotting options
    plt.xlim([data[0],data[-1]])
    plt.xlabel('Data per event (bytes)', fontsize=24)
    plt.ylabel('Overhead per batch', fontsize=24)
    plt.gca().tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, which='both')
    plt.legend(loc='lower left', prop={'size':12})
    # plt.title('Overhead with $c/s={0}$'.format(r))
    plt.savefig('mira_r{0}.pdf'.format(r))
    plt.close()

for i, r in enumerate(ratio):
    plt.semilogx(procs, [overheadB[p,r,15360] for p in procs], colors[i] + '-',
               label='$c/s={0}$, Blocking'.format(r))
    plt.semilogx(procs, [overheadNB[p,r,15360] for p in procs], colors[i] + '--',
               label='$c/s={0}$, Non-blocking'.format(r))

# plt.xlim([procs[0],procs[-1]])
plt.xlabel('Number of processors, $p$', fontsize=24)
plt.ylabel('Overhead per batch', fontsize=24)
plt.gca().set_xscale('log',basex=2)
plt.gca().set_xticklabels([str(p) for p in procs])
plt.gca().tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, which='both')
plt.legend(loc='upper right', prop={'size':12})
# plt.title('Overhead as function of $p$ with $d=15360$')
plt.savefig('mira_cs.pdf')
plt.close()

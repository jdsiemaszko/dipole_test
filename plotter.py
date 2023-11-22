import numpy as np 
import timeit
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from pathlib import Path
# import blobs_solver2 as pHyFlow
import pHyFlow
# import VorticityFoamPy as foam

import os
import sys
import yaml
import re
import csv
import pandas

suffix = ""
# PlotFlag = True
PlotFlag = False

#---------------Current directory and paths-------------------
arg = sys.argv
if len(arg) > 2:
    raise Exception("More than two arguments inserted!")
if len(arg) <= 1:
    raise Exception("No config file specificed!")
configFile = arg[1]

#-----------------------Config the yaml file ------------------
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))
config = yaml.load(open(os.path.join(configFile)),Loader=loader)

start_index = configFile.index('_') + 1
end_index = configFile.index('.')

case = config['case']

case_dir = os.path.join(os.getcwd(), 'results', case)
data_dir = os.path.join(case_dir,config["data_folder"]+suffix)
plots_dir = os.path.join(case_dir,config["plots_folder"]+suffix)

Path(plots_dir).mkdir(parents=True, exist_ok=True)

# nTimeSteps = config['nTimeSteps']
coreSize = config['coreSize']
# Gamma = config['Gamma']
Gamma = 0.0
deltaTc = config['deltaTc']
writeInterval_plots = config['writeInterval_plots']


uxNorm = np.array([])
uyNorm = np.array([])
omegaNorm = np.array([])
t_norm = np.array([])
#Line plots
times_file = os.path.join(data_dir,"times_{}.csv".format(case))
times_data = pandas.read_csv(times_file)

time = times_data['Time']
noBlobs = times_data['NoBlobs']
evolution_time = times_data['Evolution_time']
circulation = times_data['Circulation']

nTimeSteps = len(time)

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(time,noBlobs, label='No of Particles')
plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title('Total number of particles')
plt.ylabel('Particles')
plt.xlabel('time $(sec)$')
plt.legend()
plt.savefig("{}/number_of_particles_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(time,circulation- Gamma, label='Circulation deficit')
plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title('absolute error in circulation')
plt.ylabel('circulation')
plt.xlabel('time $(sec)$')
plt.legend()
plt.savefig("{}/circulation_error_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

fig = plt.subplots(figsize=(6,6))
index = np.arange(len(evolution_time))
width = deltaTc
lagrangian = plt.bar(index[1:]*deltaTc, evolution_time[1:], width)
plt.ylabel('Time (s)')
plt.xlabel('Simulation time (s)')
plt.title('Evolution time')
plt.savefig("{}/times_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

nPlotPoints = config["nPlotPoints"]
xMinPlot = config["xMinPlot"]
xMaxPlot = config["xMaxPlot"]
yMinPlot = config["yMinPlot"]
yMaxPlot = config["yMaxPlot"]

max_omega = np.empty(int(nTimeSteps / writeInterval_plots)+1)
y_boundary = np.linspace(yMinPlot, yMaxPlot, nPlotPoints)
omega_boundary = np.empty((int(nTimeSteps / writeInterval_plots)+1, y_boundary.shape[0]))

for timeStep in range(0, nTimeSteps+1):
    if timeStep%writeInterval_plots == 0:
        ####Fields
        lagrangian_file = os.path.join(data_dir,'results_{}_{n:06d}.csv'.format(case,n=timeStep))
        lagrangian_data = np.genfromtxt(lagrangian_file)

        xplot = lagrangian_data[:,0]
        yplot = lagrangian_data[:,1]
        length = int(np.sqrt(len(xplot)))
        xPlotMesh = xplot.reshape(length,length)
        yPlotMesh = yplot.reshape(length,length)

        lagrangian_ux = lagrangian_data[:,2]
        lagrangian_uy = lagrangian_data[:,3]
        lagrangian_omega = lagrangian_data[:,4]
        max_omega[int(timeStep/writeInterval_plots)] = np.max(lagrangian_omega)
        omega_boundary[int(timeStep/writeInterval_plots), :] = lagrangian_omega[np.where(xplot == -1)]

        if PlotFlag:
            xTicks = np.linspace(-2,2,5)
            yTicks = np.linspace(-2,2,5)
            fig, ax = plt.subplots(1,1,figsize=(6,6))
            ax.set_aspect("equal")
            ax.set_xticks(xTicks)
            ax.set_yticks(yTicks)
            plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cax = ax.contourf(xPlotMesh,yPlotMesh,lagrangian_omega.reshape(length,length),levels=100,cmap='RdBu',extend="both")
            cbar = fig.colorbar(cax,format="%.4f")
            cbar.set_label("Vorticity (1/s)")
            plt.tight_layout()
            plt.savefig("{}/vorticity_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            fig, ax = plt.subplots(1,1,figsize=(6,6))
            ax.set_aspect("equal")
            ax.set_xticks(xTicks)
            ax.set_yticks(yTicks)
            plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cax = ax.contourf(xPlotMesh,yPlotMesh,lagrangian_ux.reshape(length,length),levels=100,cmap='RdBu',extend="both")
            cbar = fig.colorbar(cax,format="%.4f")
            cbar.set_label("Velocity (1/s)")
            plt.tight_layout()
            plt.savefig("{}/velocity_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
            plt.close(fig)

    #### Blobs distribution

            blobs_file = os.path.join(data_dir,'blobs_{}_{n:06d}.csv'.format(case,n=timeStep))
            blobs_data = np.genfromtxt(blobs_file)

            if len(blobs_data.shape) == 1:
                blobs_x = blobs_data[0]
                blobs_y = blobs_data[1]
                blobs_g = blobs_data[2]
                if coreSize == 'variable':
                    blobs_sigma = blobs_data[3]
            else:
                blobs_x = blobs_data[:,0]
                blobs_y = blobs_data[:,1]
                blobs_g = blobs_data[:,2]
                if coreSize == 'variable':
                    blobs_sigma = blobs_data[:,3]
            if coreSize == 'variable':
                fig, ax = plt.subplots(1,1,figsize=(6,6))
                ax.scatter(blobs_x,blobs_y,c=blobs_g, s= blobs_sigma * 0.2 / np.min(blobs_sigma))
                plt.savefig("{}/blobs_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                fig, ax = plt.subplots(1,1,figsize=(6,6))
                ax.scatter(blobs_x,blobs_y,c=blobs_g, s=0.2)
                plt.savefig("{}/blobs_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
                plt.close(fig)

write_time = np.arange(0, deltaTc*(nTimeSteps+1), deltaTc*writeInterval_plots)
reference_data = np.loadtxt(os.getcwd() + "/reference_vorticity.csv", delimiter=",", dtype = np.float64, skiprows=1)
ind = np.where(write_time <= np.max(reference_data[:, 0]))[0]
fig, ax = plt.subplots(1,1,figsize=(6,6))
plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
ax.set_xlabel("Time")
ax.set_ylabel("Maximum Vorticity")
minx = np.min(xPlotMesh)
indeces = np.where(xPlotMesh == minx)[0]
plt.plot(write_time[ind], max_omega[ind], marker='x', label=case, color = 'r')
plt.plot(reference_data[:, 0], reference_data[:, 1], label='FE (reference)', color='k')
plt.legend()
plt.tight_layout()
plt.savefig("{}/max_vorticity_evolution_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1,1,figsize=(6,6))
plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
ax.set_ylabel("Y Position")
ax.set_xlabel("Boundary Vorticity")
ax.set_ybound(yMinPlot, yMaxPlot)
color = iter(cm.rainbow(np.linspace(0, 1, len(write_time))))
for i in range(len(write_time)):
    c = next(color)
    plt.plot(omega_boundary[i, :], y_boundary, label = 't={0:.3f}'.format(write_time[i]), color = c)

plt.legend()
plt.tight_layout()
plt.savefig("{}/boundary_vorticity_evolution_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")
plt.close(fig)

import os
import sys
import yaml
import pandas
import numpy as np
import csv
import re
import timeit
from pathlib import Path
import pHyFlow
from pHyFlow.blobs.base.induced import vorticity
import matplotlib.pyplot as plt

PlotFlag = False
suffix1 = ""
suffix2 = ""
#---------------Current directory and paths---------------------F---------------
arg = sys.argv
if len(arg) > 3:
    raise Exception("More than two arguments inserted!")
if len(arg) <= 2:
    raise Exception("Not enough cases to compare (need two)!")
config1File = arg[1]
config2File = arg[2]

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

config1 = yaml.load(open(os.path.join(config1File)),Loader=loader)
config2 = yaml.load(open(os.path.join(config2File)),Loader=loader)

case1 = config1['case']

case_dir1 = os.path.join(os.getcwd(), 'results', case1)
data_dir1 = os.path.join(case_dir1,config1["data_folder"]+suffix1)
plots_dir1= os.path.join(case_dir1,config1["plots_folder"]+suffix1)


case2 = config2['case']

case_dir2 = os.path.join(os.getcwd(), 'results', case2)
data_dir2 = os.path.join(case_dir2,config2["data_folder"]+suffix2)
plots_dir2 = os.path.join(case_dir2,config2["plots_folder"]+suffix2)

comp_dir = os.path.join(os.getcwd(), 'results', 'comparisons')

name_string = f'comparison_{case1}_vs_{case2}'
plots_dir =  os.path.join(comp_dir, name_string)
Path(plots_dir).mkdir(parents=True, exist_ok=True)

times_file = os.path.join(data_dir1,"times_{}.csv".format(case1))
times_ref_file = os.path.join(data_dir2,"times_{}.csv".format(case2))
times_data = pandas.read_csv(times_file)
times_ref_data = pandas.read_csv(times_ref_file)

time1 = times_data['Time']
noBlobs1 = times_data['NoBlobs']
evolution_time1 = times_data['Evolution_time']
circulation1 = times_data['Circulation']

time2 = times_ref_data['Time']
noBlobs2 = times_ref_data['NoBlobs']
evolution_time2 = times_ref_data['Evolution_time']
circulation2 = times_ref_data['Circulation']

nTimeSteps = min(len(time1), len(time2))
gammaC = circulation2[0]
deltaTc = time2[2] - time2[1]

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(time1,noBlobs1, label=case1)
ax.plot(time2,noBlobs2, label=case2)

plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title('Total number of particles')
plt.ylabel('Particles')
plt.xlabel('time $(sec)$')
plt.legend()
plt.savefig("{}/number_of_particles_{}.png".format(plots_dir,case1), dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(time1,circulation1- gammaC, label=case1)
ax.plot(time2,circulation2- gammaC, label=case2)

plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title('Circulation')
plt.ylabel('circulation')
plt.xlabel('time $(sec)$')
plt.legend()
plt.savefig("{}/circulation_{}.png".format(plots_dir,case1), dpi=300, bbox_inches="tight")

fig = plt.subplots(figsize=(6,6))
index = np.arange(len(evolution_time1))
width = deltaTc
lagrangian = plt.bar(index[1:nTimeSteps]*deltaTc, evolution_time1[1:nTimeSteps], width, label=case1, alpha=0.5)
lagrangian = plt.bar(index[1:nTimeSteps]*deltaTc, evolution_time2[1:nTimeSteps], width, label=case2, alpha=0.5)
plt.legend()
plt.ylabel('Timestep Time (s)')
plt.xlabel('Simulation time(s)')
plt.title('Evolution Times')
plt.savefig("{}/times_{}.png".format(plots_dir,case1), dpi=300, bbox_inches="tight")

fig = plt.subplots(figsize=(6,6))
index = np.arange(len(evolution_time1))
width = deltaTc
lagrangian = plt.bar(index[1:nTimeSteps]*deltaTc, evolution_time1[1:nTimeSteps] - evolution_time2[1:nTimeSteps], width)

plt.ylabel('Timestep Time (s)')
plt.xlabel('Simulation time(s)')
plt.title(f'Evolution Time Difference, {case1} - {case2}')
plt.savefig("{}/times_dif_{}.png".format(plots_dir,case1), dpi=300, bbox_inches="tight")

writeInterval_plots1 = config1["writeInterval_plots"]
writeInterval_plots2 = config2["writeInterval_plots"]

coreSize1 = config1["coreSize"]
coreSize2 = config2["coreSize"]

nPlotPoints1 = config1["nPlotPoints"]
xMinPlot1 = config1["xMinPlot"]
xMaxPlot1 = config1["xMaxPlot"]
yMinPlot1 = config1["yMinPlot"]
yMaxPlot1 = config1["yMaxPlot"]
plotting_params1 = [writeInterval_plots1, nPlotPoints1, xMinPlot1, xMaxPlot1, yMinPlot1, yMaxPlot1]

nPlotPoints2 = config2["nPlotPoints"]
xMinPlot2 = config2["xMinPlot"]
xMaxPlot2 = config2["xMaxPlot"]
yMinPlot2 = config2["yMinPlot"]
yMaxPlot2 = config2["yMaxPlot"]
plotting_params2 = [writeInterval_plots2, nPlotPoints2, xMinPlot2, xMaxPlot2, yMinPlot2, yMaxPlot2]

plotting_param_mishmatch = False
for p1, p2 in zip(plotting_params1, plotting_params2):
    if p1 != p2:
        plotting_param_mishmatch = True
        print(f'detected a mismatch in plotting params!')

time = []
L2error = []
Linferror = []
B_errorL2 = []
B_errorLinf = []
B_errorAbs = []
max_omega1 = []
max_omega2 = []

if not plotting_param_mishmatch:
    for timeStep in range(0, nTimeSteps+1):
        if timeStep%writeInterval_plots1 == 0:
            file1 = os.path.join(data_dir1,'results_{}_{n:06d}.csv'.format(case1,n=timeStep))
            data1 = np.genfromtxt(file1)

            file2 = os.path.join(data_dir2,'results_{}_{n:06d}.csv'.format(case2,n=timeStep))
            data2 = np.genfromtxt(file2)

            xplot = data1[:,0]
            yplot = data1[:,1]
            length = int(np.sqrt(len(xplot)))
            xPlotMesh = xplot.reshape(length,length)
            yPlotMesh = yplot.reshape(length,length)

            ux1 = data1[:,2]
            uy1 = data1[:,3]
            omega1 = data1[:,4]

            ux2 = data2[:,2]
            uy2 = data2[:,3]
            omega2 = data2[:,4]

            max_omega1.append(np.max(omega1))
            max_omega2.append(np.max(omega2))

            b_omega1 = omega1[np.where(xplot == -1)]
            b_omega2 = omega2[np.where(xplot == -1)]

            B_errorL2.append(np.linalg.norm(b_omega1 - b_omega2) /max(np.abs(b_omega2)))
            B_errorLinf.append(max(np.abs(b_omega1-b_omega2)) / max(np.abs(b_omega2)))
            B_errorAbs.append(max(np.abs(b_omega1 - b_omega2)))

            time.append(timeStep)
            L2error.append(np.linalg.norm(omega1-omega2) / max(np.abs(omega2)))
            Linferror.append(max(np.abs(omega1-omega2)) / max(np.abs(omega2)))


            # xTicks = np.linspace(-xMinPlot1, xMaxPlot1, 5)
            # yTicks = np.linspace(-yMinPlot1, yMaxPlot1, 5)
            if PlotFlag:
                fig, ax = plt.subplots(1,1,figsize=(6,6))
                ax.set_aspect("equal")
                # ax.set_xticks(xTicks)
                # ax.set_yticks(yTicks)
                plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                cax = ax.contourf(xPlotMesh,yPlotMesh,np.abs((omega1.reshape(length,length) - omega2.reshape(length, length))) / max(np.abs(omega2)),levels=100,cmap='RdBu',extend="both")
                cbar = fig.colorbar(cax,format="%.4f")
                cbar.set_label("Vorticity Error (%)")
                plt.tight_layout()
                plt.savefig("{}/vorticity_error_%_{}_vs_{}_{}.png".format(plots_dir,case1,case2, timeStep), dpi=300, bbox_inches="tight")
                plt.close(fig)
                
                fig, ax = plt.subplots(1,1,figsize=(6,6))
                ax.set_aspect("equal")
                # ax.set_xticks(xTicks)
                # ax.set_yticks(yTicks)
                plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
                plt.minorticks_on()
                plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                abs_vel_error = np.sqrt((ux1.reshape(length,length) - ux2.reshape(length, length))**2 + (uy1.reshape(length,length) - uy2.reshape(length, length))**2)
                vel_magnitude = np.sqrt(ux2.reshape(length,length)**2+uy2.reshape(length,length)**2)
                cax = ax.contourf(xPlotMesh,yPlotMesh,abs_vel_error / np.max(vel_magnitude),levels=100,cmap='RdBu',extend="both")
                cbar = fig.colorbar(cax,format="%.4f")
                cbar.set_label("Velocity Error (%)")
                plt.tight_layout()
                plt.savefig("{}/velocity_error_%_{}_vs_{}_{}.png".format(plots_dir,case1, case2,timeStep), dpi=300, bbox_inches="tight")
                plt.close(fig)

            

else:
    print(f'skipping vorticity and velocity plots due to plotting parameter mismatch!')
    # for timeStep in range(nTimeSteps+1):
    #     if timeStep%writeInterval_plots == 0:

    #         blobs_file = os.path.join(data_dir,'blobs_{}_{n:06d}.csv'.format(case,n=timeStep))
    #         blobs_data = np.genfromtxt(blobs_file)
    #         blobs_x = blobs_data[:,0]
    #         blobs_y = blobs_data[:,1]
    #         blobs_g = blobs_data[:,2]
    #         if coreSize == 'variable':
    #             blobs_sigma = blobs_data[:,3]
    #         else:
    #             blobs_sigma = sigma_ref
    #         # print(blobs_x, blobs_y, blobs_g, blobs_sigma)

    #         blobs_ref_file = os.path.join(ref_dir,'blobs_{}_{n:06d}.csv'.format(case_ref,n=timeStep))
    #         blobs_ref_data = np.genfromtxt(blobs_ref_file)
    #         ref_x = blobs_ref_data[:,0]
    #         ref_y = blobs_ref_data[:,1]
    #         ref_g = blobs_ref_data[:,2]
    #         if coreSize == 'variable':
    #             ref_sigma = blobs_ref_data[:,3]
    #         else:
    #             ref_sigma = sigma_ref
    #         # print(ref_x, ref_y, ref_g, ref_sigma)

    #         # print(sum(blobs_g) - sum(ref_g))


    #         omega = vorticity(np.array(blobs_x), np.array(blobs_y), np.array(blobs_g), np.array(blobs_sigma), xEval = np.array(xplotflat), yEval = np.array(yplotflat))
    #         omega_ref = vorticity(np.array(ref_x), np.array(ref_y), np.array(ref_g), np.array(ref_sigma), xEval = np.array(xplotflat), yEval = np.array(yplotflat))
    #         # print(omega, omega_ref)
    #         # print(np.linalg.norm(omega-omega_ref))
    #         # print(max(omega-omega_ref), max(np.abs(omega_ref)), max(np.abs(omega)))
    #         time.append(timeStep)
    #         L2error.append(np.linalg.norm(omega-omega_ref))
    #         Linferror.append(max(np.abs(omega-omega_ref)))
            
    #         xPlotMesh = xplotflat.reshape(length, length)
    #         yPlotMesh = yplotflat.reshape(length, length)
    #         # xTicks = np.linspace(-2,2,5)
    #         # yTicks = np.linspace(-2,2,5)

    #         fig, ax = plt.subplots(1,1,figsize=(12,6))
    #         ax.set_aspect("equal")
    #         # ax.set_xticks(xTicks)
    #         # ax.set_yticks(yTicks)
    #         plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
    #         plt.minorticks_on()
    #         plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    #         ax.set_xlabel("x")
    #         ax.set_ylabel("y")
    #         # BEWARE OF SKETCHY SOLUTION FOR LEVELS
    #         e_abs = (omega.reshape(length,length) - omega_ref.reshape(length,length))
    #         e_abs_max = np.max(np.abs(e_abs))
    #         step = e_abs_max / 50
    #         try:
    #             cax = ax.contourf(xPlotMesh,yPlotMesh, e_abs ,levels=np.arange(-e_abs_max, e_abs_max+step, step),cmap='RdBu',extend="both")
    #             cbar = fig.colorbar(cax,format="%.4f")
    #             cbar.set_label("Absolute Vorticity Error (1/s)")
    #             plt.tight_layout()
    #             plt.savefig("{}/absolute_vorticity_error_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
    #             plt.close(fig)
    #         except:
    #             pass

    #         fig, ax = plt.subplots(1,1,figsize=(12,6))
    #         ax.set_aspect("equal")
    #         # ax.set_xticks(xTicks)
    #         # ax.set_yticks(yTicks)
    #         plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
    #         plt.minorticks_on()
    #         plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    #         ax.set_xlabel("x")
    #         ax.set_ylabel("y")
    #         # BEWARE OF SKETCHY SOLUTION FOR LEVELS
    #         e_rel = (omega.reshape(length,length) - omega_ref.reshape(length,length))/ max(omega_ref) * 100
    #         e_rel_max = np.max(np.abs(e_rel))
    #         step_rel = e_rel_max/50
    #         try:
    #             cax = ax.contourf(xPlotMesh,yPlotMesh, e_rel ,levels=np.arange(-e_rel_max, e_rel_max+step_rel, step_rel),cmap='RdBu',extend="both")
    #             cbar = fig.colorbar(cax,format="%.4f")
    #             cbar.set_label("Relative Vorticity Error (%)")
    #             plt.tight_layout()
    #             plt.savefig("{}/relative_vorticity_error_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
    #             plt.close(fig)
    #         except:
    #             pass

plt.clf()
fig, ax1 = plt.subplots(figsize=(6,6))
ax1.plot(time, L2error, label='L_2', color = 'r', marker='x')
# ax2 = ax1.twinx()
# ax2.plot(time, Linferror, label='Linf', color='tab:red', marker='x')
ax1.plot(time, Linferror, label='L_inf', color='b', marker='x')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Error')
plt.legend()
plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# ax2.set_ylabel('Linf error', color='tab:red')
plt.title('Error Evolution Over Time')
plt.savefig("{}/error_evolution_{}_vs_{}.png".format(plots_dir,case1, case2), dpi=300, bbox_inches="tight")

plt.clf()
fig, ax1 = plt.subplots(figsize=(6,6))
ax1.plot(time, B_errorAbs, color = 'r', marker='x')
# ax2 = ax1.twinx()
# ax2.plot(time, Linferror, label='Linf', color='tab:red', marker='x')
# ax1.plot(time, B_errorLinf, label='L_inf', color='b', marker='x')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Error')
plt.legend()
plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# ax2.set_ylabel('Linf error', color='tab:red')
plt.title('Boundary Error Evolution Over Time (Absolute)')
plt.savefig("{}/boundary_error_evolution_{}_vs_{}.png".format(plots_dir,case1, case2), dpi=300, bbox_inches="tight")


write_time = np.arange(0, deltaTc*(nTimeSteps+1), deltaTc*writeInterval_plots1)
reference_data = np.loadtxt(os.getcwd() + "/reference_vorticity.csv", delimiter=",", dtype = np.float64, skiprows=1)

fig, ax = plt.subplots(1,1,figsize=(6,6))
plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
ax.set_xlabel("Time")
ax.set_ylabel("Maximum Vorticity")
minx = np.min(xPlotMesh)
indeces = np.where(xPlotMesh == minx)[0]
plt.plot(write_time, max_omega1, marker='x', label=case1, color = 'r')
plt.plot(write_time, max_omega2, marker='2', label=case2, color = 'b')
plt.plot(reference_data[:, 0], reference_data[:, 1], label='FE (reference)', color='k')
plt.legend()
plt.tight_layout()
plt.savefig("{}/max_vorticity_comparison_{}_vs_{}.png".format(plots_dir,case1, case2), dpi=300, bbox_inches="tight")
plt.close(fig)
""" 
--------------------------------------------
 Imbibition Code
This program calculates the relative permeability functions for an
imbibition test, using openpnm software. To run the program, you 
need to have SCAL.py in the same folder as imbibition.py. Also, 
make sure that you checked the residual saturation array vs contact 
angle and IFT.

Developed by:
    Matin Bagheri
    mail: matinbagheri2378@gmail.com
--------------------------------------------
""" 
from IPython import get_ipython

# Clear all variables
ipython = get_ipython()
if ipython:
    ipython.magic('reset -sf')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import openpnm as op
import matplotlib.pyplot as plt
import pandas as pd
import os
import inspect
import openpnm.models.geometry as gmods
import SCAL as sc

current_frame = inspect.currentframe()
script_path = inspect.getfile(current_frame)

# Get the folder name and file name
folder_name = os.path.basename(os.path.dirname(script_path))
file_name = os.path.splitext(os.path.basename(script_path))[0]

# Create the Excel file name
pc_excel_file_name = os.path.join(folder_name, f"{folder_name}_{file_name}_Pc.xlsx")
kr_excel_file_name = os.path.join(folder_name, f"{folder_name}_{file_name}_Kr.xlsx")
pcCurve_name = os.path.join(folder_name, f"{folder_name}_{file_name}_pcCurve")
KrCurve_name = os.path.join(folder_name, f"{folder_name}_{file_name}_KrCurve")
#-----------------------------------------------------------------------------------
# ______________________________________________Network Preparation______________________________________ 
#-----------------------------------------------------------------------------------

Nx, Ny, Nz, Lc = (50, 20, 10, 5e-4)

pn = op.network.BodyCenteredCubic(shape=[Nx, Ny, Nz], spacing=Lc)
np.random.seed()
pn.add_model_collection(op.models.collections.geometry.pyramids_and_cuboids)

mean = 3.8e-5
sigma = 0.75 * mean
size = pn.Np
pore_diameters = np.random.normal(mean, sigma, size)

# Assign pore diameters to the network
pn['pore.diameter'] = pore_diameters

# Generate lognormal distribution for throat sizes
mean_throat = 2.8e-6
sigma_throat = 0.25 * mean_throat
size_throat = pn.Nt
throat_diameters = np.random.normal(mean_throat, sigma_throat, size_throat)

# Assign throat diameters to the network
pn['throat.diameter'] = throat_diameters
pn.regenerate_models()


#----------------------------------------------------------------------------------------------------------
# _______________________________________________________Define phases____________________________________________________
#----------------------------------------------------------------------------------------------------------
theta_list = [0, 20, 60, 80]
gamma_list = [0.035, 0.055, 0.075]
for theta in theta_list:
    for gamma in gamma_list:
        air = op.phase.Air(network=pn,name='air')
        air['pore.contact_angle'] = 180-theta
        air['throat.contact_angle'] = 180 - theta
        air['pore.surface_tension'] = gamma
        air['throat.surface_tension'] = gamma
        air['pore.density'] = 724
        air['throat.density'] = 724
        air['pore.viscosity'] = 0.087E-3
        air['throat.viscosity'] = 0.087E-3
        f = op.models.physics.capillary_pressure.washburn
        air.add_model(propname='throat.entry_pressure',
                      model=f, 
                      surface_tension='throat.surface_tension',
                      contact_angle='throat.contact_angle',
                      diameter='throat.diameter',)

        air.add_model_collection(op.models.collections.phase.air)
        air.add_model_collection(op.models.collections.physics.basic)
        air.regenerate_models()

        water = op.phase.Water(network=pn,name='water')
        water['pore.contact_angle'] = 180 - air['pore.contact_angle']
        water['throat.contact_angle'] = 180 - air['throat.contact_angle']
        water['pore.surface_tension'] = air['pore.surface_tension']
        water['throat.surface_tension'] = air['throat.surface_tension']
        water['pore.density'] = 1053
        water['throat.density'] = 1053
        water['pore.viscosity'] = 1.087E-3
        water['throat.viscosity'] = 1.087E-3
        water.add_model(propname='throat.entry_pressure',
                      model=f, 
                      surface_tension='throat.surface_tension',
                      contact_angle='throat.contact_angle',
                      diameter='throat.diameter',)

        water.add_model_collection(op.models.collections.phase.water)
        water.add_model_collection(op.models.collections.physics.basic)
        water.regenerate_models()
        
        
        #------------------------------------------------------------------------------------
        # ______________________________________Apply Drainage and imbibition_____________________________________________
        #-----------------------------------------------------------------------------------
        injectPhase = air
        if injectPhase==air:
            Pmax = max(max(injectPhase['pore.entry_pressure']), 1)
            Prange = np.arange(0, Pmax, Pmax/100)
        else:
            Pmin = min(min(injectPhase['pore.entry_pressure']), -1)
            Prange = np.arange(Pmin, 0, abs(Pmin)/100)
        imb = op.algorithms.Drainage(network=pn, phase=air)
        Finlets_init = pn.pores('left')
        Finlets=([Finlets_init[x] for x in range(0, len(Finlets_init), 2)])
        Foutlets_init = pn.pores('right')
        Foutlets=([Foutlets_init[x] for x in range(0, len(Foutlets_init), 2)])
        imb.set_inlet_BC(pores=Finlets)  
        imb.run()
        #imb.apply_trapping()
        
        #--------------------------------------------------------------------------
        #_______________________________________Calculating Kr_______________________________________________
        #--------------------------------------------------------------------------
        
        flow_in = pn.pores('left')
        flow_out = pn.pores('right')
        
        model_mp_cond = op.models.physics.multiphase.conduit_conductance
        air.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
                      throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')
        water.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
                      throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')
        
            
        (Snwparr, relperm_wp, relperm_nwp) = sc.relperm(pn, imb, air, water, 1000, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')

        
        relperm_nwp[relperm_nwp>1] = 1
        relperm_wp[relperm_wp>1] = 1
        # Plot relative permeability curves
        plt.figure(figsize=[6, 5])
        plt.plot(Snwparr, relperm_nwp, 'r-o', label='nwp')
        plt.plot(Snwparr, relperm_wp, 'b-o', label='wp')
        plt.ylabel('relative permeability')
        plt.xlabel('Wetting Phase Saturation')
        plt.title('$\theta$ = {}, $\gamma$ = {}'.format(theta,gamma))
        plt.legend()
        plt.show()
        #plt.savefig(KrCurve_name)




""" 
--------------------------------------------
 SCAL Code
This program contains the necessary functions for an imbibition 
test, using openpnm software. To run the program, you need to 
have SCAL.py in the same folder as imbibition.py. Also, make sure 
that you checked the residual saturation array vs contact angle 
and IFT.

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
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator

def sat_occ_update(network, nwp, wp, ip, i):
    r"""
        Calculates the saturation of each phase using the invasion
        sequence from either invasion percolation.
        Parameters
        ----------
        network: network
        nwp : phase
            non-wetting phase
        wp : phase
            wetting phase
        ip : IP
            invasion percolation (ran before calling this function)
        i: int
            The invasion_sequence limit for masking pores/throats that
            have already been invaded within this limit range. The
            saturation is found by adding the volume of pores and thorats
            that meet this sequence limit divided by the bulk volume.
    """
    pore_mask = ip['pore.invasion_sequence'] < i
    throat_mask = ip['throat.invasion_sequence'] < i
    if np.mean(wp['throat.contact_angle']) > 90:
        #thetaAngle = np.pi - np.deg2rad(wp['throat.contact_angle'])
        thetaAngle = 180 - (wp['throat.contact_angle'])
    else:
        thetaAngle = (wp['throat.contact_angle'])

    #network['throat.volume'][throat_mask] = network['throat.volume'][throat_mask] * (triangleArea-3*cornerArea)/(triangleArea)
    sat_p = np.sum(network['pore.volume'][pore_mask])
    # The data here is gathered from the results of the sensitivity test.
    # For a custom geometry, run the rest again with your capillary tube
    # model and put the results here.
    thetaArray = np.array([0, 20, 40, 60, 80, 90])
    gammaArray = np.array([0.01, 0.03, 0.05, 0.07, 0.08])
    SwArr = np.array([[0.73, 0.73, 0.98, 0.98, 0.98], 
                     [0.72, 0.72, 0.98, 0.98, 0.98],
                     [0.56, 0.56, 0.98, 0.98, 0.98],
                     [0.27, 0.24, 0.23, 0.97, 0.97],
                     [0.00, 0.00, 0.00, 0.00, 0.00],
                     [0.00, 0.00, 0.00, 0.00, 0.00]])
    grid_theta, grid_gamma = np.meshgrid(thetaArray, gammaArray)
    points = np.column_stack((grid_theta.ravel(), grid_gamma.ravel()))
    satArray = 1-SwArr
    values = satArray.ravel()
    interpolator = RegularGridInterpolator((thetaArray, gammaArray), satArray)
    #interpolator = LinearNDInterpolator(points, values)
    gamma = wp['throat.surface_tension']
    if np.mean(wp['throat.contact_angle']) > 90:
        sat_t = np.sum(network['throat.volume'][throat_mask]* interpolator((thetaAngle[throat_mask], gamma[throat_mask])))
        #sat_t = np.sum(network['throat.volume'][throat_mask])
    sat1 = sat_p + sat_t  
    bulk = network['pore.volume'].sum() + network['throat.volume'].sum()
    sat = sat1/bulk
    nwp['pore.occupancy'] = pore_mask
    nwp['throat.occupancy'] = throat_mask
    wp['throat.occupancy'] = 1-throat_mask
    wp['pore.occupancy'] = 1-pore_mask
    return sat

def Rate_calc(network, phase, inlet, outlet, P, conductance):
    phase.regenerate_models()
    St_p = op.algorithms.StokesFlow(network=network, phase=phase)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=inlet, values=P)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run()
    val = np.abs(St_p.rate(pores=inlet, mode='group'))
    return val

def relperm(pn, alg, wp, nwp, P, inlet, outlet, conductance):
    """
    calculates the relative permeabilities of both phases
    
    variables:
        network: openpnm network object
        wp  : openpnm phase object - wetting phase
        nwp : openpnm phase obect - nonwetting phase
        sat : float - saturation of the wetting phase
        Pc : float - Pnw - Pw
        inlet  : inlet pores, usually network.pores('left')
        outlet : outlet pores, usually network.pores('right')
        conductance : connectivity of each phase
    
    """
    wp.regenerate_models()
    nwp.regenerate_models()
    flow_in = pn.pores('left')
    flow_out = pn.pores('right')

    model_mp_cond = op.models.physics.multiphase.conduit_conductance
    nwp.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
                  throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')
    wp.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
                  throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')

    Snwp_num=20
    flow_in = pn.pores('left')
    flow_out = pn.pores('right')
    max_seq = np.max([np.max(alg['pore.invasion_sequence'][alg['pore.invasion_sequence']!=np.inf]),
              np.max(alg['throat.invasion_sequence'][alg['throat.invasion_sequence']!=np.inf])])
    start = int(max_seq//Snwp_num)
    stop = int(max_seq)
    step = int(max_seq//Snwp_num)
    Snwparr = []
    relperm_nwp = []
    relperm_wp = []

    for i in range(start, stop, step):
        nwp.regenerate_models();
        wp.regenerate_models();
        sat = sat_occ_update(network=pn, nwp=nwp, wp=wp, ip=alg, i=i)
        Pc = np.interp(sat, alg.pc_curve().snwp, np.sort(abs(alg.pc_curve().pc)))
        Snwparr.append(sat)

        P = max(abs(wp['throat.entry_pressure']))*1
        Rate_abs_nwp = Rate_calc(pn, nwp, flow_in, flow_out, P, conductance = 'throat.hydraulic_conductance')
        Rate_abs_wp = Rate_calc(pn, wp, flow_in, flow_out, P, conductance = 'throat.hydraulic_conductance')
        Rate_enwp = Rate_calc(pn, nwp, flow_in, flow_out, P-(1-sat)*Pc, conductance = 'throat.conduit_hydraulic_conductance')
        Rate_ewp = Rate_calc(pn, wp, flow_in, flow_out, P+Pc*sat, conductance = 'throat.conduit_hydraulic_conductance')
        relperm_nwp.append(Rate_enwp/Rate_abs_nwp)
        relperm_wp.append(Rate_ewp/Rate_abs_wp)
        
    relperm_nwp = np.hstack(relperm_nwp)
    relperm_wp = np.hstack(relperm_wp)
    Snwparr = np.hstack(Snwparr)
    KrMax = max(max(relperm_nwp) , max(relperm_wp))

    
    return Snwparr, relperm_wp, relperm_nwp
    
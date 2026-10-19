"""
 Improved post-processing solution to match RTKLib behavior
"""
import numpy as np
from copy import copy, deepcopy
from rtkpos import rtkpos, rtkinit
import __ppk_config as cfg
import rinex as rn
from pntpos import pntpos
import rtkcmn as gn


def valcomb(solf, solb):
    """validation of combined solutions like RTKLib"""
    dr = np.zeros(3)
    var = np.zeros(3)
    
    # compare forward and backward solution
    for i in range(3):
        dr[i] = solf.rr[i] - solb.rr[i]
        var[i] = solf.qr[i,i] + solb.qr[i,i]  # diagonal elements
    
    for i in range(3):
        if dr[i]**2 <= 16.0 * var[i]:  # ok if in 4-sigma
            continue
        
        gn.trace(2, f"degrade fix to float: dr={dr[0]:.3f} {dr[1]:.3f} {dr[2]:.3f} "
                   f"std={np.sqrt(var[0]):.3f} {np.sqrt(var[1]):.3f} {np.sqrt(var[2]):.3f}\n")
        return False
    return True


def combres(solf, solb):
    """combine forward/backward solutions like RTKLib"""
    i, j, solc = 0, len(solb) - 1, []
    pri = [0, 1, 2, 3, 4, 5, 1, 6]  # priority of solution status (RTKLib order)
    
    gn.trace(3, f'combres: # forward = {len(solf)}, # backward = {len(solb)}\n')
    
    while i < len(solf) or j >= 0:
        if i >= len(solf):
            sol = deepcopy(solb[j])
            j -= 1
        elif j < 0:
            sol = deepcopy(solf[i])
            i += 1
        elif solf[i].stat == gn.SOLQ_NONE and solb[j].stat == gn.SOLQ_NONE:
            i += 1
            j -= 1
            continue
        else:
            tt = gn.timediff(solf[i].t, solb[j].t)
            
            if tt < -gn.DTTOL:
                sol = deepcopy(solf[i])
                i += 1
            elif tt > gn.DTTOL:
                sol = deepcopy(solb[j])
                j -= 1
            elif pri[solf[i].stat] < pri[solb[j].stat]:
                sol = deepcopy(solf[i])
                i += 1
                j -= 1
            elif pri[solf[i].stat] > pri[solb[j].stat]:
                sol = deepcopy(solb[j])
                i += 1
                j -= 1
            else:
                # Same priority - combine using smoother
                sol = deepcopy(solf[i])
                sol.t = gn.timeadd(sol.t, -tt / 2.0)
                
                # Degrade fix to float if validation failed (like RTKLib)
                if sol.stat == gn.SOLQ_FIX:
                    if not valcomb(solf[i], solb[j]):
                        sol.stat = gn.SOLQ_FLOAT
                
                # Use smoother for position combination
                Qf = np.zeros((3, 3))
                Qb = np.zeros((3, 3))
                
                # Build covariance matrices
                for k in range(3):
                    Qf[k, k] = solf[i].qr[k, k]
                    Qb[k, k] = solb[j].qr[k, k]
                
                # Off-diagonal elements
                if len(solf[i].qr) >= 6:
                    Qf[0, 1] = Qf[1, 0] = solf[i].qr[0, 1] if solf[i].qr.shape[0] > 1 else 0
                    Qf[0, 2] = Qf[2, 0] = solf[i].qr[0, 2] if solf[i].qr.shape[0] > 2 else 0
                    Qf[1, 2] = Qf[2, 1] = solf[i].qr[1, 2] if solf[i].qr.shape[0] > 2 else 0
                    
                    Qb[0, 1] = Qb[1, 0] = solb[j].qr[0, 1] if solb[j].qr.shape[0] > 1 else 0
                    Qb[0, 2] = Qb[2, 0] = solb[j].qr[0, 2] if solb[j].qr.shape[0] > 2 else 0
                    Qb[1, 2] = Qb[2, 1] = solb[j].qr[1, 2] if solb[j].qr.shape[0] > 2 else 0
                
                # Smoother combination
                sol.rr[0:3], Qs = gn.smoother(solf[i].rr[0:3], solb[j].rr[0:3], Qf, Qb)
                sol.qr = Qs
                
                i += 1
                j -= 1
        
        solc.append(sol)
    
    return solc


def firstpos(nav, rov, base, dir):
    """find rover position from first obs with better error handling"""
    max_attempts = 10
    attempts = 0
    
    while attempts < max_attempts:
        obsr, obsb = rn.first_obs(nav, rov, base, dir)
        if not obsr or len(obsr.sat) == 0:
            break
            
        sol = pntpos(obsr, nav)
        if sol.stat != gn.SOLQ_NONE:
            gn.trace(3, f'init rr: {sol.rr[0]:.2f} {sol.rr[1]:.2f} {sol.rr[2]:.2f}: {sol.stat}\n')
            nav.x[0:6] = copy(sol.rr[0:6])
            nav.rr[0:3] = copy(sol.rr[0:3])
            return
        
        # Try next observation
        obsr, obsb = rn.next_obs(nav, rov, base, dir)
        attempts += 1
    
    # Fallback to approximate position
    gn.trace(2, 'Warning: Could not get initial position from observations\n')
    nav.x[0:3] = [0, 0, 0]  # Will be set later


def sqrtvar(cov):
    """sqrt of covariance with better handling"""
    return np.sqrt(np.abs(cov)) * np.sign(cov)


def savesol(sol, solfile):
    """Save solution with RTKLib-compatible format"""
    D2R = gn.rCST.D2R
    solhdr = '%  GPST                  latitude(deg) longitude(deg)  height(m)   Q  ' \
        'ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio\n'
    
    with open(solfile, 'w') as outfile:
        outfile.write(solhdr)
        
        # Remove duplicates based on time
        processed_times = set()
        
        for s in sol:
            if s.stat == gn.SOLQ_NONE:
                continue
                
            # Create unique time identifier
            time_id = f"{s.t.time}_{s.t.sec:.6f}"
            if time_id in processed_times:
                continue
            processed_times.add(time_id)
            
            # Convert to standard date/time format like RTKLib
            ep = gn.time2epoch(s.t)
            llh = gn.ecef2pos(s.rr[0:3])
            
            # Handle covariance matrix properly
            if s.qr.shape == (3, 3):
                std = sqrtvar(gn.covenu(llh, s.qr))
            else:
                # Fallback for incorrect shape
                std = np.zeros((3, 3))
                for i in range(min(3, len(s.qr))):
                    std[i, i] = np.sqrt(abs(s.qr[i]))
            
            # Format time like RTKLib: YYYY/MM/DD HH:MM:SS.SSS
            time_str = f"{ep[0]:04d}/{ep[1]:02d}/{ep[2]:02d} {ep[3]:02d}:{ep[4]:02d}:{ep[5]:06.3f}"
            
            fmt = '%s %14.9f %14.9f %10.4f %3d %3d %8.4f' + \
                '  %8.4f %8.4f %8.4f %8.4f %8.4f %6.2f %6.1f\n'
            
            outfile.write(fmt % (
                time_str, 
                llh[0]/D2R, llh[1]/D2R, llh[2], 
                s.stat, s.ns, 
                std[1,1], std[0,0], std[2,2], 
                std[0,1], std[2,0], std[1,2], 
                s.age, s.ratio
            ))


def procpos(nav, rov, base, fp_stat):
    """Main processing function with RTKLib-style logic"""
    try:
        if nav.filtertype != 'backward':
            # Forward solution
            firstpos(nav, rov, base, dir=1)
            rtkpos(nav, rov, base, fp_stat, dir=1) 
            sol0 = deepcopy(nav.sol)
            savesol(sol0, 'forward.pos')
            
        if nav.filtertype != 'forward':
            # Backward solution
            if nav.filtertype != 'combined_noreset':
                # Reset filter states like RTKLib
                rb = nav.rb.copy()
                eph, geph = nav.eph.copy(), nav.geph.copy()
                glofrq = nav.glofrq.copy()
                maxepoch = nav.maxepoch
                nav = rtkinit(cfg)
                nav.rb = rb
                nav.eph, nav.geph = eph, geph
                nav.glofrq = glofrq
                nav.maxepoch = maxepoch
                firstpos(nav, rov, base, dir=-1)
            else:
                # Combined_noreset: keep last solution
                if nav.sol:
                    nav.sol = [nav.sol[-1]]
            
            rtkpos(nav, rov, base, fp_stat, dir=-1)  
            savesol(nav.sol, 'backward.pos')
            
        if nav.filtertype == 'combined' or nav.filtertype == 'combined_noreset':
            # Combine forward and backward solutions
            sol = combres(sol0, nav.sol)
            savesol(sol, 'combined.pos')
            return sol
            
    except KeyboardInterrupt:
        gn.trace(1, "Processing interrupted by user\n")
        pass
    except Exception as e:
        gn.trace(1, f"Processing error: {str(e)}\n")
        pass
    
    return nav.sol
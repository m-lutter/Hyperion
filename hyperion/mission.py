"""Mission analysis kernels using SI units and explicit, supplied assumptions.

TMI continues the shared 6DOF model. Lambert/separation are explicitly two-body
screens, not ephemeris generation, collision certification, or a window optimum.
Reference: https://docs.poliastro.space/en/stable/autoapi/poliastro/iod/vallado/
"""
import copy
import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from .trajectory import Simulation, load_config, run_trajectory, matrix, unit, MU, RE, G0


def _positive(value, name, allow_zero=False):
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}.')
    return float(value)


def _vector(value, name):
    value=np.asarray(value,dtype=float)
    if value.shape!=(3,) or not np.isfinite(value).all():
        raise ValueError(f'{name} must contain three finite SI values.')
    return value


def state_from_row(row):
    """Reconstruct all 22 integrated states, including attitude and loss integrals."""
    names=([f'eci_{k}_m' for k in 'xyz']+[f'eci_v{k}_m_s' for k in 'xyz']+
           [f'quaternion_{k}' for k in 'xyzw'])
    state=[float(row[k]) for k in names]
    state+=list(np.deg2rad([row[f'body_rate_{k}_deg_s'] for k in 'xyz']))
    state += [row['prop1_kg'],row['prop2_kg']]
    state += list(np.deg2rad([row['gimbal_y_deg'],row['gimbal_z_deg']]))
    state += [row[k] for k in ['ideal_dv_m_s','steering_loss_m_s','gimbal_loss_m_s',
                              'drag_loss_m_s','gravity_loss_m_s']]
    result=np.asarray(state,dtype=float)
    if not np.isfinite(result).all(): raise ValueError('Cutoff state contains nonfinite values.')
    return result


class _DepartureSimulation(Simulation):
    def guidance(self,t,r,vrel,v,h,qbar):
        # Prograde steering is an explicit feasible-policy screen, not targeting
        # optimization. The original actuator/6DOF response still follows it.
        return unit(v)


def finite_burn_tmi(*, cutoff_row, config, coast_duration_s, restart_propellant_kg,
                    target_c3_km2_s2, max_burn_duration_s, output_step_s=1.0):
    """Continue an ascent-cutoff state through coast, restart, and finite TMI.

    Required coast/restart values are not silently invented. C3 is the Earth
    two-body characteristic energy, not proof of Mars arrival geometry. Thermal
    boiloff and ignition transients require separate supplied models. The cutoff
    row must describe the detached, fairing-free upper stage; its mass and CG are
    checked against the supplied configuration before any phase transition.
    """
    _positive(coast_duration_s,'coast_duration_s',True)
    _positive(restart_propellant_kg,'restart_propellant_kg',True)
    _positive(max_burn_duration_s,'max_burn_duration_s')
    _positive(output_step_s,'output_step_s')
    if not math.isfinite(target_c3_km2_s2): raise ValueError('Target C3 must be finite.')
    c=copy.deepcopy(config)
    c['solver']['output_step_s']=output_step_s
    c['solver']['verbose']=False
    sim=_DepartureSimulation(c)
    sim.phase=3; sim.attached=False; sim.fairing=False; sim.cutoff=True
    y=state_from_row(cutoff_row); t=float(cutoff_row['t_s']); start=t
    if 'mass_kg' not in cutoff_row or 'cg_from_base_m' not in cutoff_row:
        raise ValueError('Cutoff row must include mass_kg and cg_from_base_m to verify vehicle configuration.')
    supplied=np.array([cutoff_row['mass_kg'],cutoff_row['cg_from_base_m']],dtype=float)
    def matches(fairing):
        expected=np.asarray(sim.vehicle.properties(y[13:15],False,fairing)[:2])
        return bool(np.all(np.isfinite(supplied)) and np.allclose(supplied,expected,rtol=1e-8,atol=1e-6))
    if not matches(False):
        if matches(True):
            raise ValueError('Cutoff row retains the fairing; supply an explicit fairing-jettison state before TMI.')
        raise ValueError('Cutoff mass/CG is incompatible with the detached, fairing-free upper-stage configuration.')
    initial_prop=float(y[14]); initial_ideal=float(y[17])
    def energy(state): return (np.dot(state[3:6],state[3:6])-2*MU/np.linalg.norm(state[:3]))/1e6
    def ground(time,state): return np.linalg.norm(state[:3])-RE
    ground.terminal=True; ground.direction=-1
    sim.events=[{'event':'ascent_cutoff','t_s':t}]
    status='coast_complete'
    if coast_duration_s:
        y,t,sol=sim.integrate(y,t,t+coast_duration_s,[ground])
        if sol.t_events[0].size: status='ground_contact'
    residual=c['vehicle']['stages'][1]['residual_kg']
    if status!='ground_contact':
        # An already-satisfied energy target requires no ignition or startup debit.
        if energy(y)>=target_c3_km2_s2:
            status='C3_already_at_or_above_target'
        elif y[14]-restart_propellant_kg <= residual:
            status='insufficient_restart_inventory'
        else:
            before=sim.vehicle.properties(y[13:15],False,False)[1]
            y[14]-=restart_propellant_kg
            after=sim.vehicle.properties(y[13:15],False,False)[1]
            shift=matrix(y[6:10])@np.array([after-before,0.,0.])
            y[:3]+=shift; y[3:6]+=np.cross(matrix(y[6:10])@y[10:13],shift)
            sim.phase=2; sim.cutoff=False; sim.ignition_time=t
            sim.events.append({'event':'TMI_restart','t_s':t,'startup_propellant_kg':restart_propellant_kg})
            def depleted(time,state): return state[14]-residual
            depleted.terminal=True; depleted.direction=-1
            def target(time,state): return energy(state)-target_c3_km2_s2
            target.terminal=True; target.direction=1
            y,t,sol=sim.integrate(y,t,t+max_burn_duration_s,[depleted,target,ground])
            status=('residual_propellant_limit' if sol.t_events[0].size else
                    'target_C3_reached' if sol.t_events[1].size else
                    'ground_contact' if sol.t_events[2].size else 'burn_time_limit')
    sim.events.append({'event':status,'t_s':t})
    # integrate() retains the powered left-limit at the shutdown event for load
    # checks. Append a separate engine-off right-limit without changing the state.
    sim.phase=3; sim.cutoff=True
    sim.rows.append(sim.evaluate(t,y,True))
    history=pd.DataFrame(sim.rows)
    final_c3=float(energy(y))
    powered=history[history.thrust_N>0]
    return {'status':status,'target_C3_reached':bool(final_c3>=target_c3_km2_s2-1e-7),
            'target_C3_km2_s2':float(target_c3_km2_s2),'achieved_C3_km2_s2':final_c3,
            'C3_margin_km2_s2':final_c3-target_c3_km2_s2,
            'elapsed_from_ascent_cutoff_s':t-start,'remaining_propellant_kg':float(y[14]),
            'required_residual_kg':float(residual),'consumed_propellant_kg':initial_prop-float(y[14]),
            'TMI_ideal_delta_v_m_s':float(y[17]-initial_ideal),'events':sim.events,
            'maximum_axial_g':float(history.axial_g.abs().max()),
            'maximum_lateral_g':float(history.lateral_g.abs().max()),
            'axial_acceleration_screen_pass':bool(history.axial_g.abs().max()<=c['limits']['axial_g']),
            'lateral_acceleration_screen_pass':bool(history.lateral_g.abs().max()<=c['limits']['lateral_g']),
            'powered_tracking_capacity_screen_pass':(None if powered.empty else
                bool((powered.required_moment_Nm<=powered.available_moment_Nm+1e-6).all())),
            'history':history.to_dict(orient='records'),
            'assumptions':['Same shared 6DOF dynamics, atmosphere, moving mass, actuator and acceleration limits.',
                           'Prograde departure guidance; no optimizer or planetary encounter matching.',
                           'Supplied instantaneous restart propellant debit; no ignition shock or boiloff model.',
                           'Source gimbal hardware and coast-control limitations remain applicable.']}


def integrated_tmi(*, coast_duration_s, restart_propellant_kg, target_c3_km2_s2,
                   max_burn_duration_s, config=None, ascent_history=None, output_step_s=1.0,
                   context=None):
    """Reuse or run the ascent, then continue from its actual orbital-speed cutoff."""
    if context is not None:
        ascent_history,_,simulation=context
        config=simulation.c
    c=copy.deepcopy(load_config() if config is None else config)
    if ascent_history is None:
        frame,summary,sim=run_trajectory(c)
        if not any(e['event']=='orbital_speed_cutoff' for e in sim.events):
            return {'status':'ascent_did_not_reach_cutoff','target_C3_reached':False,'ascent_summary':summary}
    else:
        frame=pd.DataFrame(ascent_history)
        summary=None
    powered=frame[(frame.phase==2)&(frame.thrust_N>0)]
    if powered.empty: raise ValueError('Ascent history contains no powered upper-stage state.')
    row=powered.iloc[-1].to_dict()
    if not (frame[(frame.t_s>=row['t_s'])&(frame.phase==3)]).shape[0]:
        raise ValueError('Supplied ascent history does not demonstrate an orbital-cutoff coast.')
    result=finite_burn_tmi(cutoff_row=row,config=c,coast_duration_s=coast_duration_s,
                          restart_propellant_kg=restart_propellant_kg,target_c3_km2_s2=target_c3_km2_s2,
                          max_burn_duration_s=max_burn_duration_s,output_step_s=output_step_s)
    if summary is not None: result['ascent_summary']=summary
    return result


def _stumpff(z):
    if abs(z)<1e-6:
        return .5-z/24+z*z/720-z**3/40320,1/6-z/120+z*z/5040-z**3/362880
    x=math.sqrt(abs(z))
    if z>0: return (1-math.cos(x))/z,(x-math.sin(x))/x**3
    return (math.cosh(x)-1)/(-z),(math.sinh(x)-x)/x**3


def lambert(*, departure_position_m, arrival_position_m, time_of_flight_s,
            central_mu_m3_s2, long_way=False):
    """Zero-revolution universal-variable solution; endpoints share one inertial frame.

    `long_way` chooses transfer angle > pi. Collinear endpoints are rejected;
    they require a specified transfer plane and a different singularity treatment.
    """
    r1=_vector(departure_position_m,'departure_position_m')
    r2=_vector(arrival_position_m,'arrival_position_m')
    mu=_positive(central_mu_m3_s2,'central_mu_m3_s2'); tof=_positive(time_of_flight_s,'time_of_flight_s')
    n1=np.linalg.norm(r1); n2=np.linalg.norm(r2)
    _positive(n1,'departure radius'); _positive(n2,'arrival radius')
    cos_angle=float(np.clip(np.dot(r1,r2)/(n1*n2),-1,1))
    sin_angle=math.sqrt(max(0.,1-cos_angle*cos_angle))*(-1 if long_way else 1)
    if abs(sin_angle)<1e-9: raise ValueError('Collinear Lambert endpoints require a specified transfer plane.')
    A=sin_angle*math.sqrt(n1*n2/(1-cos_angle))
    def geometry(z):
        C,S=_stumpff(z)
        if C<=0: return None
        y=n1+n2+A*(z*S-1)/math.sqrt(C)
        if y<0: return None
        elapsed=((y/C)**1.5*S+A*math.sqrt(y))/math.sqrt(mu)
        return elapsed,y
    # The zero-revolution branch ends at the first positive Stumpff singularity.
    grid=np.r_[-np.geomspace(1e4,1e-8,700),0.,np.linspace(1e-8,4*math.pi**2-1e-6,700)]
    previous=None; bracket=None
    for z in grid:
        value=geometry(float(z))
        if value is None: continue
        f=value[0]-tof
        if previous is not None and previous[1]*f<=0:
            bracket=(previous[0],float(z)); break
        previous=(float(z),f)
    if bracket is None: raise ValueError('No zero-revolution Lambert root bracketed for these endpoints/time.')
    z=brentq(lambda z:geometry(z)[0]-tof,*bracket,xtol=1e-12,rtol=1e-13)
    elapsed,y=geometry(z)
    f=1-y/n1; g=A*math.sqrt(y/mu); gdot=1-y/n2
    v1=(r2-f*r1)/g; v2=(gdot*r2-r1)/g
    return {'departure_velocity_m_s':v1.tolist(),'arrival_velocity_m_s':v2.tolist(),
            'time_of_flight_residual_s':float(elapsed-tof),'universal_parameter':float(z),
            'branch':'zero-revolution long-way' if long_way else 'zero-revolution short-way'}


def lambert_window(*, cases, central_mu_m3_s2):
    """Evaluate supplied ephemeris/date pairs; no fabricated planet ephemerides.

    Each case includes endpoint positions, planet velocities, flight time, and
    optional date labels. The caller must supply a common inertial frame/epoch.
    Declination is relative to the supplied XY plane; use equatorial ICRF states.
    """
    output=[]
    if not cases: raise ValueError('At least one ephemeris case is required.')
    for case in cases:
        result=lambert(departure_position_m=case['departure_position_m'],arrival_position_m=case['arrival_position_m'],
                       time_of_flight_s=case['time_of_flight_s'],central_mu_m3_s2=central_mu_m3_s2,
                       long_way=case.get('long_way',False))
        departure_excess=np.asarray(result['departure_velocity_m_s'])-_vector(case['departure_planet_velocity_m_s'],'departure_planet_velocity_m_s')
        arrival_excess=np.asarray(result['arrival_velocity_m_s'])-_vector(case['arrival_planet_velocity_m_s'],'arrival_planet_velocity_m_s')
        result.update({'departure_label':case.get('departure_label','supplied epoch'),
                       'arrival_label':case.get('arrival_label','supplied epoch'),
                       'departure_C3_km2_s2':float(np.dot(departure_excess,departure_excess)/1e6),
                       'departure_excess_velocity_m_s':departure_excess.tolist(),
                       'departure_declination_deg':float(np.degrees(np.arctan2(departure_excess[2],np.linalg.norm(departure_excess[:2])))),
                       'arrival_excess_speed_m_s':float(np.linalg.norm(arrival_excess))})
        output.append(result)
    return {'cases':output,'minimum_C3_case_index':int(np.argmin([x['departure_C3_km2_s2'] for x in output])),
            'scope':'Minimum among supplied samples only; not a global launch-window optimization.'}


def relative_separation(*, position_a_m, velocity_a_m_s, position_b_m, velocity_b_m_s,
                        mass_a_kg, mass_b_kg, impulse_on_a_N_s, duration_s,
                        required_center_clearance_m, central_mu_m3_s2=MU, sample_step_s=.1):
    """Propagate both released bodies with equal/opposite impulse and gravity.

    The supplied clearance is a center-distance keep-out envelope; body rotation,
    plumes, flex, contact dynamics, and ignition forces are not included.
    """
    m1=_positive(mass_a_kg,'mass_a_kg'); m2=_positive(mass_b_kg,'mass_b_kg')
    duration=_positive(duration_s,'duration_s'); dt=_positive(sample_step_s,'sample_step_s')
    mu=_positive(central_mu_m3_s2,'central_mu_m3_s2'); clearance=_positive(required_center_clearance_m,'required_center_clearance_m',True)
    impulse=_vector(impulse_on_a_N_s,'impulse_on_a_N_s')
    p1=_vector(position_a_m,'position_a_m');p2=_vector(position_b_m,'position_b_m')
    if np.linalg.norm(p1)<1 or np.linalg.norm(p2)<1: raise ValueError('Position cannot be at the central-body singularity.')
    v1=_vector(velocity_a_m_s,'velocity_a_m_s')+impulse/m1
    v2=_vector(velocity_b_m_s,'velocity_b_m_s')-impulse/m2
    initial=np.r_[p1,v1,p2,v2]
    def rhs(t,y):
        return np.r_[y[3:6],-mu*y[:3]/np.linalg.norm(y[:3])**3,
                     y[9:12],-mu*y[6:9]/np.linalg.norm(y[6:9])**3]
    ts=np.unique(np.r_[np.arange(0,duration,dt),duration])
    sol=solve_ivp(rhs,(0,duration),initial,t_eval=ts,rtol=1e-10,atol=1e-8,method='RK45',max_step=dt)
    if not sol.success: raise RuntimeError(sol.message)
    dr=sol.y[:3].T-sol.y[6:9].T; dv=sol.y[3:6].T-sol.y[9:12].T
    distance=np.linalg.norm(dr,axis=1); k=int(np.argmin(distance))
    return {'time_s':sol.t.tolist(),'center_distance_m':distance.tolist(),
            'relative_position_m':dr.tolist(),'relative_velocity_m_s':dv.tolist(),
            'minimum_center_distance_m':float(distance[k]),'minimum_time_s':float(sol.t[k]),
            'required_center_clearance_m':clearance,'sampled_clearance_pass':bool(np.all(distance>=clearance)),
            'delta_v_a_m_s':(impulse/m1).tolist(),'delta_v_b_m_s':(-impulse/m2).tolist(),
            'scope':'Two point masses with supplied center-distance envelope; sampled check only.'}


def rocket_equation(*, initial_mass_kg, final_mass_kg, isp_s):
    m0=_positive(initial_mass_kg,'initial_mass_kg'); mf=_positive(final_mass_kg,'final_mass_kg')
    isp=_positive(isp_s,'isp_s')
    if mf>m0: raise ValueError('Final mass cannot exceed initial mass.')
    return {'ideal_delta_v_m_s':G0*isp*math.log(m0/mf),'propellant_consumed_kg':m0-mf,
            'scope':'Ideal rocket equation; gravity, steering, drag and startup losses excluded.'}


def propellant_budget(*, loaded_mass_kg, burns_kg, startup_kg, boiloff_kg, trapped_residual_kg, required_reserve_kg):
    values=[loaded_mass_kg,startup_kg,boiloff_kg,trapped_residual_kg,required_reserve_kg,*burns_kg]
    for v in values:_positive(v,'inventory term',True)
    remaining=loaded_mass_kg-sum(burns_kg)-startup_kg-boiloff_kg
    available=remaining-trapped_residual_kg
    return {'remaining_total_kg':float(remaining),'usable_after_residual_kg':float(available),
            'reserve_margin_kg':float(available-required_reserve_kg),'reserve_pass':bool(available>=required_reserve_kg)}


def payload_sweep(*, payload_masses_kg, config=None, context=None):
    """Rerun the shared ascent for each supplied payload; no mass rescaling shortcut."""
    if not payload_masses_kg: raise ValueError('At least one payload mass is required.')
    if context is not None:config=context[2].c
    baseline=load_config() if config is None else config; cases=[]
    for payload in payload_masses_kg:
        _positive(payload,'payload mass')
        c=copy.deepcopy(baseline);c['vehicle']['payload_kg']=float(payload);c['solver']['verbose']=False
        frame,summary,sim=run_trajectory(c)
        cases.append({'payload_kg':float(payload),'summary':summary})
    return {'cases':cases,'scope':'Ascent capability at supplied masses; not finite-TMI payload optimization.'}


def finite_tmi_payload_sweep(*,payload_masses_kg,coast_duration_s,restart_propellant_kg,
                             target_c3_km2_s2,max_burn_duration_s,config=None,context=None):
    """Repeat both shared 6DOF ascent and departure for each requested payload mass."""
    if not payload_masses_kg:raise ValueError('At least one payload mass is required.')
    baseline=context[2].c if context is not None else load_config() if config is None else config
    cases=[]
    for mass in payload_masses_kg:
        _positive(mass,'payload mass')
        c=copy.deepcopy(baseline);c['vehicle']['payload_kg']=float(mass);c['solver']['verbose']=False
        result=integrated_tmi(config=c,coast_duration_s=coast_duration_s,restart_propellant_kg=restart_propellant_kg,
            target_c3_km2_s2=target_c3_km2_s2,max_burn_duration_s=max_burn_duration_s)
        result.pop('history',None);result['payload_kg']=float(mass);cases.append(result)
    return {'cases':cases,'scope':'Full reruns at supplied masses and prescribed coast/prograde guidance; not a global optimum.'}


def tmi_cutoff_sensitivity(*,cutoff_row,config,burn_durations_s,coast_duration_s,
                           restart_propellant_kg,target_c3_km2_s2):
    """Sweep prescribed TMI shutdown times, still stopping at propellant residual."""
    if not burn_durations_s:raise ValueError('At least one burn duration is required.')
    cases=[]
    for duration in burn_durations_s:
        _positive(duration,'burn duration')
        # Disable energy cutoff for this timing study, never disable residual protection.
        result=finite_burn_tmi(cutoff_row=cutoff_row,config=config,coast_duration_s=coast_duration_s,
            restart_propellant_kg=restart_propellant_kg,target_c3_km2_s2=1e12,max_burn_duration_s=duration)
        cases.append({'commanded_burn_duration_s':float(duration),'status':result['status'],
            'achieved_C3_km2_s2':result['achieved_C3_km2_s2'],
            'C3_margin_km2_s2':result['achieved_C3_km2_s2']-target_c3_km2_s2,
            'remaining_propellant_kg':result['remaining_propellant_kg']})
    return {'target_C3_km2_s2':float(target_c3_km2_s2),'cases':cases,
            'scope':'Prescribed cutoff-time sensitivity; no inferred timing uncertainty or optimized guidance.'}


def separation_monte_carlo(*,samples,seed,position_covariance_m2,velocity_covariance_m2,
                           impulse_covariance_N2_s2,separation_inputs):
    """Sample supplied joint state/impulse covariance; record invalid cases as failures."""
    if int(samples)!=samples or samples<1:raise ValueError('samples must be a positive integer.')
    def covariance(value,size,name):
        a=np.asarray(value,dtype=float)
        if a.shape!=(size,size) or not np.isfinite(a).all() or not np.allclose(a,a.T,atol=1e-12):
            raise ValueError(f'{name} must be a finite symmetric {size}x{size} matrix.')
        if np.linalg.eigvalsh(a).min() < -1e-12:raise ValueError(f'{name} must be positive semidefinite.')
        return a
    pc=covariance(position_covariance_m2,6,'position covariance')
    vc=covariance(velocity_covariance_m2,6,'velocity covariance')
    jc=covariance(impulse_covariance_N2_s2,3,'impulse covariance')
    rng=np.random.default_rng(seed);cases=[]
    for i in range(int(samples)):
        inp=copy.deepcopy(separation_inputs)
        dp=rng.multivariate_normal(np.zeros(6),pc);dv=rng.multivariate_normal(np.zeros(6),vc)
        dj=rng.multivariate_normal(np.zeros(3),jc)
        for index,body in enumerate(['a','b']):
            inp[f'position_{body}_m']=(np.asarray(inp[f'position_{body}_m'])+dp[3*index:3*index+3]).tolist()
            inp[f'velocity_{body}_m_s']=(np.asarray(inp[f'velocity_{body}_m_s'])+dv[3*index:3*index+3]).tolist()
        inp['impulse_on_a_N_s']=(np.asarray(inp['impulse_on_a_N_s'])+dj).tolist()
        try:
            result=relative_separation(**inp)
            cases.append({'sample':i,'minimum_center_distance_m':result['minimum_center_distance_m'],
                          'sampled_clearance_pass':result['sampled_clearance_pass']})
        except (ValueError,RuntimeError) as error:
            cases.append({'sample':i,'sampled_clearance_pass':False,'error':str(error)})
    failures=sum(not x['sampled_clearance_pass'] for x in cases)
    return {'seed':int(seed),'samples':int(samples),'failed_samples':int(failures),
            'sample_failure_fraction':failures/int(samples),'cases':cases,
            'scope':'Supplied Gaussian dispersions and point-mass keep-out screen; sample fraction is not a certified failure probability.'}


FUNCTIONS={'integrated_tmi':integrated_tmi,'finite_burn_tmi':finite_burn_tmi,'lambert':lambert,
           'lambert_window':lambert_window,'relative_separation':relative_separation,
           'rocket_equation':rocket_equation,'propellant_budget':propellant_budget,'payload_sweep':payload_sweep,
           'finite_tmi_payload_sweep':finite_tmi_payload_sweep,'tmi_cutoff_sensitivity':tmi_cutoff_sensitivity,
           'separation_monte_carlo':separation_monte_carlo}
KERNEL_ACTIVITIES={
    'Integrated trajectory simulation from launch to TMI.':'integrated_tmi',
    'Integrated ascent, coast and finite-burn TMI simulation.':'integrated_tmi',
    'Integrated finite-burn trajectory simulation.':'integrated_tmi',
    'Earth-Mars Lambert launch-window sweep.':'lambert_window',
    'Spacecraft separation relative-motion simulation.':'relative_separation',
    'Stage-separation relative-motion simulation.':'relative_separation',
    'Relative-stage trajectory propagation.':'relative_separation',
    'Stage–spacecraft relative-trajectory propagation.':'relative_separation',
    'Stage rocket-equation delta-v calculations.':'rocket_equation',
    'Rocket-equation stage and propellant sizing.':'rocket_equation',
    'Burn-by-burn propellant and reserve balance.':'propellant_budget',
    'Payload-capability sensitivity sweep.':'payload_sweep',
    'Payload-capability and mass-growth sensitivity sweep.':'payload_sweep',
    'Payload-capability sweep using finite-burn trajectory simulation.':'finite_tmi_payload_sweep',
    'Finite-burn TMI cutoff sensitivity sweep.':'tmi_cutoff_sensitivity',
    'Monte Carlo spacecraft separation relative-motion simulation.':'separation_monte_carlo',
    'Monte Carlo stage-separation relative-motion simulation.':'separation_monte_carlo',
}

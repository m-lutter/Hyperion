"""Explicit-input propulsion screening calculations; no hardware test claims.

SI throughout. Relations: NASA Glenn rocket thrust, mass flow and isentropic
flow equations: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/rocket-thrust/
https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/mass-flow-rate-equations/
"""
import math
import numpy as np
from scipy.integrate import solve_ivp, trapezoid, cumulative_trapezoid
from scipy.optimize import brentq
from .trajectory import G0


def _positive(value,name,zero=False):
    value=float(value)
    if not math.isfinite(value) or (value<0 if zero else value<=0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}.')
    return value


def _history(time_s,**values):
    t=np.asarray(time_s,dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Time must contain at least two finite strictly increasing samples.')
    result={}
    for key,value in values.items():
        a=np.asarray(value,dtype=float)
        if a.ndim==0:a=np.full_like(t,a)
        if a.shape!=t.shape or not np.isfinite(a).all(): raise ValueError(f'{key} must match time samples.')
        result[key]=a
    return t,result


def nozzle(*,chamber_pressure_Pa,chamber_temperature_K,gas_constant_J_kg_K,
           heat_capacity_ratio,throat_area_m2,exit_area_m2,ambient_pressure_Pa,
           velocity_efficiency):
    """Calorically perfect isentropic nozzle; chamber chemistry/pump cycle excluded."""
    pc=_positive(chamber_pressure_Pa,'chamber pressure');tc=_positive(chamber_temperature_K,'chamber temperature')
    R=_positive(gas_constant_J_kg_K,'gas constant');g=_positive(heat_capacity_ratio,'heat capacity ratio')
    at=_positive(throat_area_m2,'throat area');ae=_positive(exit_area_m2,'exit area')
    pa=_positive(ambient_pressure_Pa,'ambient pressure',True);eta=_positive(velocity_efficiency,'velocity efficiency')
    if g<=1 or eta>1 or ae<at: raise ValueError('Require gamma > 1, efficiency <= 1, exit area >= throat area.')
    expansion=ae/at
    def area_ratio(M):return ((2/(g+1)*(1+(g-1)*M*M/2))**((g+1)/(2*(g-1))))/M
    me=1. if expansion==1 else brentq(lambda M:area_ratio(M)-expansion,1+1e-9,50.)
    ratio=1+(g-1)*me*me/2;te=tc/ratio;pe=pc/ratio**(g/(g-1))
    mdot=pc*at/math.sqrt(tc)*math.sqrt(g/R)*(2/(g+1))**((g+1)/(2*(g-1)))
    ve=eta*me*math.sqrt(g*R*te);pressure_thrust=(pe-pa)*ae;thrust=mdot*ve+pressure_thrust
    return {'exit_mach':float(me),'exit_temperature_K':te,'exit_pressure_Pa':pe,
            'exit_velocity_m_s':ve,'mass_flow_kg_s':mdot,'momentum_thrust_N':mdot*ve,
            'pressure_thrust_N':pressure_thrust,'thrust_N':thrust,'isp_s':thrust/(mdot*G0),
            'scope':'Ideal supersonic nozzle branch; separation/shocks/combustion equilibrium and full engine cycle unresolved.'}


def thrust_sum(*,per_engine_thrust_N,engine_count):
    f=_positive(per_engine_thrust_N,'per-engine thrust',True)
    if int(engine_count)!=engine_count or engine_count<1:raise ValueError('Engine count must be a positive integer.')
    return {'installed_thrust_N':f*int(engine_count),'engine_count':int(engine_count)}


def thrust_to_weight(*,installed_thrust_N,liftoff_mass_kg,local_gravity_m_s2):
    thrust=_positive(installed_thrust_N,'installed thrust',True);mass=_positive(liftoff_mass_kg,'liftoff mass')
    gravity=_positive(local_gravity_m_s2,'local gravity')
    return {'thrust_to_weight':thrust/(mass*gravity),'thrust_minus_weight_N':thrust-mass*gravity}


def mass_fraction(*,propellant_mass_kg,wet_stage_mass_kg):
    prop=_positive(propellant_mass_kg,'propellant mass',True);wet=_positive(wet_stage_mass_kg,'wet stage mass')
    if prop>wet:raise ValueError('Propellant exceeds wet stage mass.')
    return {'propellant_to_wet_mass':prop/wet,'nonpropellant_mass_kg':wet-prop}


def thrust_flow(*,thrust_N,isp_s):
    return {'mass_flow_kg_s':_positive(thrust_N,'thrust',True)/(_positive(isp_s,'specific impulse')*G0)}


def integrate_consumption(*,time_s,thrust_N,isp_s,oxidizer_fuel_ratio=None):
    """Integrate supplied piecewise-linear thrust/flow; no constant-Isp shortcut."""
    t,x=_history(time_s,thrust_N=thrust_N,isp_s=isp_s)
    if np.any(x['thrust_N']<0) or np.any(x['isp_s']<=0):raise ValueError('Require nonnegative thrust and positive Isp.')
    mdot=x['thrust_N']/(G0*x['isp_s'])
    mass=float(trapezoid(mdot,t));impulse=float(trapezoid(x['thrust_N'],t))
    result={'consumed_propellant_kg':mass,'total_impulse_N_s':impulse,'mass_flow_kg_s':mdot.tolist()}
    if oxidizer_fuel_ratio is not None:
        _,rat=_history(t,ratio=oxidizer_fuel_ratio);ratio=rat['ratio']
        if np.any(ratio<=0):raise ValueError('Mixture ratio must be positive.')
        result.update(fuel_consumed_kg=float(trapezoid(mdot/(1+ratio),t)),
                      oxidizer_consumed_kg=float(trapezoid(mdot*ratio/(1+ratio),t)))
    return result


def throttle_response(*,command_time_s,command_fraction,time_constant_s,delay_s,
                      slew_limit_per_s,initial_fraction,full_thrust_N,sample_step_s):
    """First-order response with zero-order-held command, delay, and rate limit."""
    t,x=_history(command_time_s,command=command_fraction);u=x['command']
    tau=_positive(time_constant_s,'time constant');delay=_positive(delay_s,'delay',True)
    rate=_positive(slew_limit_per_s,'slew limit');dt=_positive(sample_step_s,'sample step')
    thrust=_positive(full_thrust_N,'full thrust',True)
    if np.any((u<0)|(u>1)) or not 0<=initial_fraction<=1:raise ValueError('Throttle fractions must lie in [0,1].')
    def command(now):
        if now-delay<t[0]:return float(initial_fraction)
        return float(u[min(len(u)-1,max(0,np.searchsorted(t,now-delay,side='right')-1))])
    # Impulse is an integrated state, independent of requested display sampling.
    def rhs(now,y):return [np.clip((command(now)-y[0])/tau,-rate,rate),thrust*y[0]]
    times=np.unique(np.r_[np.arange(t[0],t[-1],dt),t[-1]])
    sol=solve_ivp(rhs,(t[0],t[-1]),[initial_fraction,0.],method='RK45',t_eval=times,
                  max_step=min(dt,tau/10),rtol=1e-8,atol=1e-10)
    if not sol.success:raise RuntimeError(sol.message)
    actual=np.clip(sol.y[0],0,1)
    return {'time_s':times.tolist(),'command_fraction':[command(v) for v in times],
            'actual_fraction':actual.tolist(),'thrust_N':(thrust*actual).tolist(),
            'total_impulse_N_s':float(sol.y[1,-1]),
            'scope':'Supplied actuator response parameters; no measured hot-fire dynamics or acceptance test.'}


def hydraulic_network(*,mass_flow_kg_s,density_kg_m3,dynamic_viscosity_Pa_s,
                      segments,upstream_pressure_Pa,elevation_rise_m,gravity_m_s2):
    """Series incompressible Darcy-Weisbach + local-loss screen at supplied flow."""
    flow=_positive(mass_flow_kg_s,'mass flow',True);rho=_positive(density_kg_m3,'density')
    viscosity=_positive(dynamic_viscosity_Pa_s,'viscosity');p=_positive(upstream_pressure_Pa,'upstream pressure',True)
    g=_positive(gravity_m_s2,'gravity',True)
    if not math.isfinite(elevation_rise_m) or not segments:raise ValueError('Supply finite elevation and at least one segment.')
    total=0.;rows=[]
    for segment in segments:
        D=_positive(segment['diameter_m'],'diameter');L=_positive(segment['length_m'],'length',True)
        rough=_positive(segment['roughness_m'],'roughness',True);K=_positive(segment['minor_loss_coefficient'],'minor loss',True)
        velocity=flow/(rho*math.pi*D*D/4);Re=rho*velocity*D/viscosity
        if Re==0:f=0.;regime='zero flow'
        elif Re<2300:f=64/Re;regime='laminar'
        else:
            f=.25/math.log10(rough/(3.7*D)+5.74/Re**.9)**2
            regime='transition (uncertain)' if Re<4000 else 'turbulent'
        loss=(f*L/D+K)*rho*velocity*velocity/2;total+=loss
        rows.append({'velocity_m_s':velocity,'reynolds_number':Re,'darcy_friction_factor':f,
                     'pressure_loss_Pa':loss,'regime':regime})
    hydro=rho*g*float(elevation_rise_m);down=p-total-hydro
    return {'segments':rows,'friction_minor_loss_Pa':total,'hydrostatic_pressure_change_Pa':hydro,
            'downstream_pressure_Pa':down,'negative_absolute_pressure_flag':bool(down<0),
            'scope':'Single-phase incompressible series network; phase change, cavitation and compressible purge not modeled.'}


def loading_time(*,target_mass_kg,initial_mass_kg,delivery_mass_flow_kg_s,loss_mass_flow_kg_s):
    target=_positive(target_mass_kg,'target mass',True);initial=_positive(initial_mass_kg,'initial mass',True)
    flow=_positive(delivery_mass_flow_kg_s,'delivery flow',True);loss=_positive(loss_mass_flow_kg_s,'loss flow',True)
    if target<initial:raise ValueError('Use drain model for target mass below initial mass.')
    if flow<=loss and target>initial:raise ValueError('Delivery must exceed losses to reach target.')
    time=0. if target==initial else (target-initial)/(flow-loss)
    return {'fill_time_s':time,'delivered_mass_kg':flow*time,'lost_mass_kg':loss*time,
            'net_fill_rate_kg_s':flow-loss}


def liquid_ullage(*,tank_volume_m3,loaded_liquid_mass_kg,liquid_density_kg_m3):
    volume=_positive(tank_volume_m3,'tank volume');mass=_positive(loaded_liquid_mass_kg,'loaded liquid mass',True)
    liquid=mass/_positive(liquid_density_kg_m3,'liquid density');ullage=volume-liquid
    return {'liquid_volume_m3':liquid,'ullage_volume_m3':ullage,'ullage_fraction':ullage/volume,
            'geometric_fill_pass':bool(ullage>=0)}


def pressurant_inventory(*,tank_pressure_Pa,initial_ullage_m3,final_ullage_m3,
                        ullage_temperature_K,gas_constant_J_kg_K,storage_volume_m3,
                        storage_initial_pressure_Pa,storage_final_pressure_Pa,
                        storage_initial_temperature_K,storage_final_temperature_K):
    """Ideal-gas, constant-pressure/isothermal receiving ullage inventory screen."""
    p=_positive(tank_pressure_Pa,'tank pressure');vi=_positive(initial_ullage_m3,'initial ullage')
    vf=_positive(final_ullage_m3,'final ullage');R=_positive(gas_constant_J_kg_K,'gas constant')
    T=_positive(ullage_temperature_K,'ullage temperature');V=_positive(storage_volume_m3,'storage volume')
    pi=_positive(storage_initial_pressure_Pa,'initial storage pressure');pf=_positive(storage_final_pressure_Pa,'residual storage pressure')
    ti=_positive(storage_initial_temperature_K,'initial storage temperature');tf=_positive(storage_final_temperature_K,'final storage temperature')
    if vf<vi:raise ValueError('Final ullage must be at least initial ullage for depletion pressurization.')
    needed=p*(vf-vi)/(R*T);initial=pi*V/(R*ti);residual=pf*V/(R*tf);deliverable=initial-residual
    pressure_margin=min(pi,pf)-p
    return {'required_added_pressurant_kg':needed,'storage_initial_mass_kg':initial,
            'storage_residual_mass_kg':residual,'deliverable_storage_mass_kg':deliverable,
            'mass_margin_kg':deliverable-needed,'mass_inventory_pass':bool(deliverable>=needed),
            'minimum_storage_to_tank_pressure_margin_Pa':pressure_margin,
            'delivery_pressure_screen_pass':bool(pressure_margin>0),
            'scope':'Mass inventory and necessary positive delivery-pressure margin are separate screens, not an overall delivery pass. Ideal gas; regulator minimum differential, real-gas correction, thermal response and line losses unresolved.'}


def ullage_transient(*,time_s,ullage_volume_m3,inflow_kg_s,outflow_kg_s,heat_input_W,
                     inlet_temperature_K,initial_mass_kg,initial_temperature_K,
                     gas_constant_J_kg_K,heat_capacity_cv_J_kg_K):
    """Lumped ideal-gas open-system d(m cv T)/dt = Q + min cp Tin - mout cp T - p dV/dt."""
    t,x=_history(time_s,V=ullage_volume_m3,minflow=inflow_kg_s,mout=outflow_kg_s,Q=heat_input_W,Tin=inlet_temperature_K)
    if np.any(x['V']<=0) or np.any(x['Tin']<=0) or np.any(x['minflow']<0) or np.any(x['mout']<0):raise ValueError('Volumes/temperatures must be positive; flows nonnegative.')
    m0=_positive(initial_mass_kg,'initial mass');T0=_positive(initial_temperature_K,'initial temperature')
    R=_positive(gas_constant_J_kg_K,'gas constant');cv=_positive(heat_capacity_cv_J_kg_K,'cv');cp=cv+R
    def rhs(now,y):
        mass,U=y;T=U/(mass*cv);V=np.interp(now,t,x['V']);p=mass*R*T/V
        k=min(len(t)-2,max(0,np.searchsorted(t,now)-1));dV=(x['V'][k+1]-x['V'][k])/(t[k+1]-t[k])
        inf=np.interp(now,t,x['minflow']);out=np.interp(now,t,x['mout'])
        return [inf-out,np.interp(now,t,x['Q'])+inf*cp*np.interp(now,t,x['Tin'])-out*cp*T-p*dV]
    def depleted(now,y):return min(y[0],y[1])
    depleted.terminal=True;depleted.direction=-1
    sol=solve_ivp(rhs,(t[0],t[-1]),[m0,m0*cv*T0],t_eval=t,events=depleted,method='RK45',
                  max_step=float(np.min(np.diff(t)))/5,rtol=1e-8,atol=1e-9)
    if not sol.success:raise RuntimeError(sol.message)
    if sol.t_events[0].size:raise ValueError('Supplied history exhausts gas mass or internal energy.')
    temperature=sol.y[1]/(sol.y[0]*cv);pressure=sol.y[0]*R*temperature/x['V']
    return {'time_s':t.tolist(),'gas_mass_kg':sol.y[0].tolist(),'temperature_K':temperature.tolist(),
            'pressure_Pa':pressure.tolist(),'scope':'Prescribed inflow/outflow, ideal-gas lumped energy balance; regulator control law not identified.'}


def boiloff_inventory(*,initial_liquid_mass_kg,heat_leak_W,latent_heat_J_kg,duration_s,replenishment_kg_s):
    """Saturated-liquid heat leak converted to boiloff; no sensible-heating assumption hidden."""
    mass=_positive(initial_liquid_mass_kg,'initial liquid mass',True);heat=_positive(heat_leak_W,'heat leak',True)
    latent=_positive(latent_heat_J_kg,'latent heat');dt=_positive(duration_s,'duration',True)
    inflow=_positive(replenishment_kg_s,'replenishment flow',True);rate=heat/latent
    final=mass+(inflow-rate)*dt
    if final<0:raise ValueError('Assumed saturated-liquid inventory is depleted before the end of the interval.')
    return {'boiloff_rate_kg_s':rate,'boiloff_mass_kg':rate*dt,'replenishment_mass_kg':inflow*dt,
            'final_liquid_mass_kg':final,'scope':'Saturated-liquid constant-property heat/mass balance; vent capacity and pressure dynamics separate.'}


def trapped_inventory(*,trapped_volumes_m3,liquid_density_kg_m3,cutoff_flow_kg_s,cutoff_delay_s):
    rho=_positive(liquid_density_kg_m3,'density');flow=_positive(cutoff_flow_kg_s,'cutoff flow',True)
    delay=_positive(cutoff_delay_s,'cutoff delay',True)
    volume=sum(_positive(v,'trapped volume',True) for v in trapped_volumes_m3)
    return {'trapped_liquid_mass_kg':rho*volume,'post_command_consumption_kg':flow*delay,
            'combined_inventory_allowance_kg':rho*volume+flow*delay}


def chilldown(*,wall_mass_kg,wall_heat_capacity_J_kg_K,initial_wall_temperature_K,
              coolant_temperature_K,heat_transfer_coefficient_W_m2_K,area_m2,
              duration_s,coolant_latent_heat_J_kg,sample_count=101):
    """Lumped wall cooling to a prescribed coolant bath, with heat-equivalent evaporation."""
    mass=_positive(wall_mass_kg,'wall mass');cp=_positive(wall_heat_capacity_J_kg_K,'wall heat capacity')
    Ti=_positive(initial_wall_temperature_K,'initial temperature');Tc=_positive(coolant_temperature_K,'coolant temperature')
    h=_positive(heat_transfer_coefficient_W_m2_K,'heat transfer coefficient');A=_positive(area_m2,'area')
    duration=_positive(duration_s,'duration');latent=_positive(coolant_latent_heat_J_kg,'latent heat')
    if sample_count<2 or int(sample_count)!=sample_count:raise ValueError('At least two samples required.')
    if Ti<Tc:raise ValueError('Chilldown requires wall initially at or above coolant temperature.')
    tau=mass*cp/(h*A);t=np.linspace(0,duration,int(sample_count));T=Tc+(Ti-Tc)*np.exp(-t/tau)
    heat=mass*cp*(Ti-T[-1])
    return {'time_s':t.tolist(),'wall_temperature_K':T.tolist(),'time_constant_s':tau,
            'removed_heat_J':float(heat),'heat_equivalent_evaporated_mass_kg':float(heat/latent),
            'scope':'Lumped fixed-coefficient thermal screen; two-phase flow and minimum operational temperature require supplied criteria.'}


def tank_geometry(*,inner_radius_m,cylinder_length_m,dome_height_m,dome_count,dome_shape,
                  loaded_liquid_mass_kg,liquid_density_kg_m3):
    """Cylinder plus explicitly selected ellipsoidal or spherical-cap dome geometry."""
    radius=_positive(inner_radius_m,'inner radius');length=_positive(cylinder_length_m,'cylinder length',True)
    height=_positive(dome_height_m,'dome height',True)
    if int(dome_count)!=dome_count or dome_count<0:raise ValueError('Dome count must be a nonnegative integer.')
    if dome_shape=='ellipsoidal':dome=2*math.pi*radius*radius*height/3
    elif dome_shape=='spherical_cap':dome=math.pi*height*(3*radius*radius+height*height)/6
    else:raise ValueError('dome_shape must be ellipsoidal or spherical_cap.')
    cylinder=math.pi*radius*radius*length;volume=cylinder+int(dome_count)*dome
    result=liquid_ullage(tank_volume_m3=volume,loaded_liquid_mass_kg=loaded_liquid_mass_kg,
                        liquid_density_kg_m3=liquid_density_kg_m3)
    return dict(result,tank_volume_m3=volume,cylinder_volume_m3=cylinder,
                per_dome_volume_m3=dome,dome_shape=dome_shape)


def feed_transient(*,time_s,mass_flow_kg_s,tank_gas_pressure_Pa,axial_acceleration_m_s2,
                   initial_liquid_mass_kg,tank_cross_section_m2,density_kg_m3,
                   dynamic_viscosity_Pa_s,vapor_pressure_Pa,segments):
    """Depleting liquid head, prescribed feed flow inertia, and series pressure loss.

    The flow history is supplied, not solved from valve/turbopump characteristics.
    Single-phase incompressible flow is invalid when the vapor margin is negative.
    """
    t,x=_history(time_s,flow=mass_flow_kg_s,pressure=tank_gas_pressure_Pa,acceleration=axial_acceleration_m_s2)
    if np.any(x['flow']<0) or np.any(x['pressure']<=0) or np.any(x['acceleration']<0):raise ValueError('Require nonnegative flow/axial acceleration and positive tank pressure.')
    initial=_positive(initial_liquid_mass_kg,'initial inventory');area=_positive(tank_cross_section_m2,'tank cross section')
    rho=_positive(density_kg_m3,'density');pv=_positive(vapor_pressure_Pa,'vapor pressure',True)
    mass=initial-cumulative_trapezoid(x['flow'],t,initial=0)
    if np.any(mass<0):raise ValueError('Prescribed feed history depletes the supplied inventory.')
    head=mass/(rho*area);derivative=np.gradient(x['flow'],t)
    # L/A * dm_dot/dt is the acceleration pressure of a constant-area liquid slug.
    inertance=sum(_positive(s['length_m'],'segment length',True)/
                 (math.pi*_positive(s['diameter_m'],'diameter')**2/4) for s in segments)
    downstream=[];losses=[];inertial=[]
    for i in range(len(t)):
        upstream=x['pressure'][i]+rho*x['acceleration'][i]*head[i]
        result=hydraulic_network(mass_flow_kg_s=x['flow'][i],density_kg_m3=rho,
            dynamic_viscosity_Pa_s=dynamic_viscosity_Pa_s,segments=segments,
            upstream_pressure_Pa=upstream,elevation_rise_m=0,gravity_m_s2=0)
        acceleration_pressure=inertance*derivative[i]
        downstream.append(result['downstream_pressure_Pa']-acceleration_pressure)
        losses.append(result['friction_minor_loss_Pa']);inertial.append(acceleration_pressure)
    pressure=np.asarray(downstream);margin=pressure-pv
    return {'time_s':t.tolist(),'remaining_liquid_mass_kg':mass.tolist(),'liquid_head_m':head.tolist(),
            'feed_inlet_pressure_Pa':pressure.tolist(),'friction_loss_Pa':losses,'inertial_pressure_Pa':inertial,
            'vapor_pressure_margin_Pa':margin.tolist(),'single_phase_pressure_screen_pass':bool(np.all(margin>0)),
            'scope':'Prescribed-flow liquid inertia/depletion screen; valve dynamics, pump suction requirements and two-phase transients unresolved.'}


def mixture_flow_transient(*,time_s,total_command_kg_s,command_oxidizer_fuel_ratio,
                           fuel_time_constant_s,oxidizer_time_constant_s,
                           initial_fuel_flow_kg_s,initial_oxidizer_flow_kg_s):
    """Two first-order feed responses to a common total-flow and mixture command."""
    t,x=_history(time_s,total=total_command_kg_s,ratio=command_oxidizer_fuel_ratio)
    if np.any(x['total']<0) or np.any(x['ratio']<=0):raise ValueError('Total flow must be nonnegative and mixture ratio positive.')
    tf=_positive(fuel_time_constant_s,'fuel time constant');to=_positive(oxidizer_time_constant_s,'oxidizer time constant')
    fi=_positive(initial_fuel_flow_kg_s,'initial fuel flow',True);oi=_positive(initial_oxidizer_flow_kg_s,'initial oxidizer flow',True)
    def rhs(now,y):
        total=np.interp(now,t,x['total']);ratio=np.interp(now,t,x['ratio'])
        return [(total/(1+ratio)-y[0])/tf,(total*ratio/(1+ratio)-y[1])/to,y[0],y[1]]
    # Cumulative fluid masses use the adaptive solver, not output-grid quadrature.
    sol=solve_ivp(rhs,(t[0],t[-1]),[fi,oi,0.,0.],t_eval=t,method='RK45',rtol=1e-8,atol=1e-10,
                  max_step=min(tf,to,float(np.min(np.diff(t))))/10)
    if not sol.success:raise RuntimeError(sol.message)
    fuel,ox=sol.y[:2];ratio=[float(o/f) if f>1e-12 else None for f,o in zip(fuel,ox)]
    return {'time_s':t.tolist(),'fuel_flow_kg_s':fuel.tolist(),'oxidizer_flow_kg_s':ox.tolist(),
            'actual_oxidizer_fuel_ratio':ratio,'fuel_consumed_kg':float(sol.y[2,-1]),
            'oxidizer_consumed_kg':float(sol.y[3,-1]),
            'scope':'Common-command coupling with supplied time constants; not a coupled combustion/pressure-control model or hot-fire validation.'}


def shutdown_surge(*,density_kg_m3,wave_speed_m_s,initial_flow_velocity_m_s,final_flow_velocity_m_s,
                   line_length_m,closure_time_s,initial_pressure_Pa):
    """Joukowsky rapid-closure pressure-rise bound with applicability flag."""
    rho=_positive(density_kg_m3,'density');wave=_positive(wave_speed_m_s,'wave speed')
    vi=_positive(initial_flow_velocity_m_s,'initial velocity',True);vf=_positive(final_flow_velocity_m_s,'final velocity',True)
    L=_positive(line_length_m,'line length');dt=_positive(closure_time_s,'closure time')
    pressure=_positive(initial_pressure_Pa,'initial pressure',True)
    rise=rho*wave*(vi-vf);transit=2*L/wave
    return {'pressure_change_Pa':rise,'bounded_pressure_Pa':pressure+rise,
            'round_trip_wave_time_s':transit,'rapid_closure_assumption_satisfied':bool(dt<=transit),
            'scope':'Single Joukowsky-wave bound; no distributed reflections, cavitation or coupled thrust-decay simulation.'}


def fill_hydraulics(*,target_mass_kg,initial_mass_kg,delivery_mass_flow_kg_s,loss_mass_flow_kg_s,
                    density_kg_m3,dynamic_viscosity_Pa_s,segments,upstream_pressure_Pa,
                    elevation_rise_m,gravity_m_s2):
    """Combine constant-rate loading inventory with pressure needed at that same rate."""
    fill=loading_time(target_mass_kg=target_mass_kg,initial_mass_kg=initial_mass_kg,
        delivery_mass_flow_kg_s=delivery_mass_flow_kg_s,loss_mass_flow_kg_s=loss_mass_flow_kg_s)
    network=hydraulic_network(mass_flow_kg_s=delivery_mass_flow_kg_s,density_kg_m3=density_kg_m3,
        dynamic_viscosity_Pa_s=dynamic_viscosity_Pa_s,segments=segments,
        upstream_pressure_Pa=upstream_pressure_Pa,elevation_rise_m=elevation_rise_m,gravity_m_s2=gravity_m_s2)
    return {'loading':fill,'hydraulics':network,
            'scope':'Prescribed constant liquid loading rate; verify source capacity, tank backpressure and cavitation separately.'}


FUNCTIONS={name:globals()[name] for name in ['nozzle','thrust_sum','thrust_to_weight','mass_fraction','thrust_flow',
    'integrate_consumption','throttle_response','hydraulic_network','loading_time','liquid_ullage',
    'pressurant_inventory','ullage_transient','boiloff_inventory','trapped_inventory','chilldown',
    'tank_geometry','feed_transient','mixture_flow_transient','shutdown_surge','fill_hydraulics']}
KERNEL_ACTIVITIES={
    'First-stage engine thrust summation.':'thrust_sum','Upper-stage engine thrust summation.':'thrust_sum',
    'Sea-level thrust-to-weight calculation.':'thrust_to_weight',
    'Installed-thrust and liftoff-weight balance.':'thrust_to_weight',
    'First-stage propellant-to-wet-mass calculation.':'mass_fraction','Upper-stage propellant-to-wet-mass calculation.':'mass_fraction',
    'Vacuum thrust-to-propellant-flow calculation.':'thrust_flow',
    'Thrust-impulse and propellant-flow integration.':'integrate_consumption',
    'First-stage propellant-consumption integration.':'integrate_consumption',
    'Upper-stage burn/coast propellant-consumption integration.':'integrate_consumption',
    'Engine throttle step-response simulation.':'throttle_response','Propulsion command-response dynamics simulation.':'throttle_response',
    'Per-engine feed-network pressure-loss calculations.':'hydraulic_network',
    'First-stage oxygen loading pressure-loss calculations.':'hydraulic_network',
    'Upper-stage oxygen loading pressure-loss calculations.':'hydraulic_network',
    'Methane loading pressure-loss calculations.':'hydraulic_network',
    'Cold-flow pressure-loss and resource-budget calculations.':'hydraulic_network',
    'Ullage and loaded-liquid volume calculations.':'liquid_ullage',
    'Pressurant consumption and storage-residual mass balance.':'pressurant_inventory',
    'Tank pressurization mass-and-energy transient simulation.':'ullage_transient',
    'Flight tank heat-and-mass-balance simulation.':'boiloff_inventory',
    'Ground tank heat-and-mass-balance simulation.':'boiloff_inventory',
    'Tank heat-balance and replenishment-flow calculations.':'boiloff_inventory',
    'Propellant inventory and boil-off mass-balance calculations.':'boiloff_inventory',
    'Trapped-volume and cutoff inventory calculation.':'trapped_inventory',
    'Chilldown transient heat-balance calculations.':'chilldown',
    'Feedline and engine chilldown transient simulation.':'chilldown',
    'Tank geometry and liquid-volume calculations.':'tank_geometry',
    'Depletion-dependent feed-network transient simulation.':'feed_transient',
    'Feed-network inlet-pressure transient simulation.':'feed_transient',
    'Coupled fuel–oxidizer flow transient simulation.':'mixture_flow_transient',
    'Fill-time and hydraulic pressure-loss calculations.':'fill_hydraulics',
}

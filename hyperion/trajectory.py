"""Hyperion 6-DOF screening model. SI units; body +x points from engine to nose.

The integrated attitude is a unit quaternion [x,y,z,w], body -> ECI. Dynamics
are continuous inside each flight phase; mass shedding is handled separately.
"""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "trajectory.json"


def load_config(path=None):
    """Return a fresh editable input dictionary; the default is config/trajectory.json.

    Aerodynamic file paths in the configuration are relative to the repository
    root unless an absolute path is supplied. The input file is never modified.
    """
    return json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8-sig"))


G0 = 9.80665
MU = 3.986004418e14
RE = 6371000.0
OMEGA = np.array([0., 0., 7.292115e-5])

def unit(v):
    return v / max(np.linalg.norm(v), 1e-15)

def matrix(q):
    x,y,z,w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def quaternion_derivative(q,w):
    return .5*np.r_[q[3]*w + np.cross(q[:3],w), -np.dot(q[:3],w)]

def local_frame(r):
    up = unit(r)
    east = unit(np.cross([0.,0.,1.],up))
    north = np.cross(up,east)
    return east,north,up

class Atmosphere:
    """US Standard Atmosphere 1976 lower layers; explicit upper extension.

    Uses geopotential height, hydrostatic pressure and ideal gas. Above 86 km
    a continuous isothermal exponential tail is a SCREENING approximation.
    Wind is a configurable deterministic profile, not a launch-day forecast.
    """
    def __init__(self,c):
        self.c=c
        self.h=np.array([0,11000,20000,32000,47000,51000,71000,84852.])
        self.lapse=np.array([-.0065,0,.001,.0028,0,-.0028,-.002])
        self.tb=[288.15]; self.pb=[101325.]
        for i,L in enumerate(self.lapse):
            dh=self.h[i+1]-self.h[i]; T=self.tb[-1]
            self.tb.append(T+L*dh)
            self.pb.append(self.pb[-1]*(np.exp(-G0*dh/(287.05287*T)) if L==0 else
                         ((T+L*dh)/T)**(-G0/(287.05287*L))))

    def sample(self,h):
        z=max(0.,h); H=6356766*z/(6356766+z)
        if H <= self.h[-1]:
            k=min(6,max(0,np.searchsorted(self.h,H,side='right')-1))
            L=self.lapse[k]; T0=self.tb[k]; dh=H-self.h[k]
            T=T0+L*dh
            p=self.pb[k]*(np.exp(-G0*dh/(287.05287*T0)) if L==0 else
                         (T/T0)**(-G0/(287.05287*L)))
        else:
            T=self.tb[-1]
            p=self.pb[-1]*np.exp(-G0*(H-self.h[-1])/(287.05287*T))
        T += self.c.get('temperature_offset_K',0.)
        rho=p/(287.05287*T)*self.c.get('density_scale',1.)
        a=np.sqrt(1.4*287.05287*T)
        viscosity=1.716e-5*(T/273.15)**1.5*(273.15+110.4)/(T+110.4)
        return T,p,rho,a,viscosity

    def wind(self,h,t,r):
        c=self.c; east,north,_=local_frame(r)
        sp=c['wind_reference_m_s']*(max(h,0.)/c['wind_reference_height_m'])**c['wind_exponent']
        sp=min(sp,c['wind_reference_m_s']*(c['wind_cap_height_m']/c['wind_reference_height_m'])**c['wind_exponent'])
        # Taper above the user-selected atmosphere region; avoid permanent wind in space.
        sp*=np.clip((c['wind_zero_height_m']-h)/max(1.,c['wind_zero_height_m']-c['wind_taper_height_m']),0,1)
        direction=np.deg2rad(c['wind_toward_deg_east_of_north'])
        gust=c.get('gust_amplitude_m_s',0.)*np.sin(2*np.pi*t/c.get('gust_period_s',12.))
        return (sp+gust)*np.cos(direction)*north+(sp+gust)*np.sin(direction)*east

class AeroTable:
    def __init__(self,path,c):
        self.c=c
        a=np.genfromtxt(path,delimiter=',',names=True,encoding='utf-8-sig')
        self.alpha=np.unique(a['alpha_deg'])
        self.cols={}
        self.mins=[]; self.maxs=[]
        for angle in self.alpha:
            rows=a[a['alpha_deg']==angle]
            rows=np.sort(rows,order='mach')
            self.cols[angle]=rows
            self.mins.append(rows['mach'].min()); self.maxs.append(rows['mach'].max())
        self.mlo=max(self.mins); self.mhi=min(self.maxs)

    def sample(self,mach,alpha,powered):
        deg=abs(np.rad2deg(alpha)); off=(mach<self.mlo or mach>self.mhi or deg>self.alpha[-1])
        # Endpoint holding is always flagged in output; never claimed as interpolation.
        mm=np.clip(mach,self.mlo,self.mhi); aa=np.clip(deg,self.alpha[0],self.alpha[-1])
        out=[]
        for col in ['ca_on' if powered else 'ca_off','cn','cp_from_nose_m']:
            vals=[np.interp(mm,self.cols[x]['mach'],self.cols[x][col]) for x in self.alpha]
            out.append(float(np.interp(aa,self.alpha,vals)))
        # CN must vanish at zero AoA for the symmetric rigid-body extension.
        if deg<1e-10: out[1]=0.
        return *out,off

class Vehicle:
    def __init__(self,c):
        self.c=c
        self.s=c['stages']
        self.offset=[0.,self.s[0]['length_m']]

    def properties(self,props,attached,fairing):
        components=[]; rad=self.c['diameter_m']/2
        for i,s in enumerate(self.s):
            if i==0 and not attached: continue
            off=self.offset[i]
            dryL=s['length_m']-(self.c['fairing_length_m'] if i==1 else 0.)
            components.append((s['dry_mass_kg'],off+.5*dryL,dryL,rad))
            frac=np.clip(props[i]/s['propellant_kg'],0.,1.)
            for tank in s['tanks']:
                # Equivalent constant-area liquid column drains downward. Slosh omitted.
                liquidL=tank['height_m']*frac
                components.append((tank['loaded_mass_kg']*frac,off+tank['bottom_m']+.5*liquidL,liquidL,rad))
        base=self.offset[1]+self.s[1]['length_m']-self.c['fairing_length_m']
        components.append((self.c['payload_kg'],base+self.c['payload_cg_above_fairing_base_m'],self.c['payload_length_m'],self.c['payload_radius_m']))
        if fairing:
            components.append((self.c['fairing_kg'],base+self.c['fairing_cg_above_base_m'],self.c['fairing_length_m'],rad))
        total=sum(x[0] for x in components)
        cg=sum(m*x for m,x,L,r in components)/total
        # Equivalent uniform solid cylinders + parallel-axis theorem.
        ix=sum(.5*m*r*r for m,x,L,r in components)
        it=sum(m*(3*r*r+L*L)/12+m*(x-cg)**2 for m,x,L,r in components)
        return total,cg,np.array([ix,it,it])

class Simulation:
    """One fresh 6DOF ascent run, with phase events and a 22-component state.

    ``root`` optionally overrides the base for relative aerodynamic-data paths.
    The configuration is copied so a run cannot modify caller-owned inputs.
    Create a new Simulation for each run; this object retains its sampled rows.
    """
    def __init__(self,c=None,root=None):
        c=copy.deepcopy(load_config() if c is None else c)
        self.c=c; self.root=Path(root) if root is not None else REPOSITORY_ROOT
        self.atm=Atmosphere(c['atmosphere'])
        if c['solver'].get('method','RK45')!='RK45': raise ValueError('This simulator requires method=RK45.')
        if c['solver']['max_time_s']<=0: raise ValueError('max_time_s must be positive.')
        for st in c['vehicle']['stages']:
            if not 0<=st['residual_kg']<st['propellant_kg']: raise ValueError('Propellant must exceed residual inventory.')
            if not 0<st['min_throttle']<=1: raise ValueError('Minimum throttle must be in (0,1].')
        self.vehicle=Vehicle(c['vehicle']); self.table=AeroTable(self.root/c['aero']['file'],c['aero'])
        self.events=[]; self.rows=[]; self.phase=1; self.attached=True; self.fairing=True
        self.cutoff=False; self.ignition_time=0.

    def initial(self):
        lat,lon=np.deg2rad([self.c['launch']['latitude_deg'],self.c['launch']['longitude_deg']])
        props=[s['propellant_kg'] for s in self.c['vehicle']['stages']]
        props[0]-=self.c['vehicle']['stages'][0]['startup_propellant_kg']
        initial_cg=self.vehicle.properties(props,True,True)[1]
        r=(RE+self.c['launch']['altitude_m']+initial_cg)*np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
        e,n,u=local_frame(r)
        q=Rotation.from_matrix(np.column_stack([u,e,n])).as_quat()
        # Start at release with launch vehicle co-rotating with the Earth.
        return np.r_[r,np.cross(OMEGA,r),q,OMEGA@matrix(q),props,0.,0.,0.,0.,0.,0.,0.]

    def guidance(self,t,r,vrel,v,h,qbar):
        c=self.c['guidance']; e,n,u=local_frame(r)
        az=np.deg2rad(self.c['launch']['azimuth_deg']); horizontal=np.sin(az)*e+np.cos(az)*n
        pitch=np.deg2rad(np.interp(t,c['pitch_time_s'],c['pitch_deg']))
        if self.phase==2:
            # Altitude hold with radial-velocity damping while building orbital speed.
            vr=np.dot(v,u); vt=np.linalg.norm(v-vr*u)
            desired_ar=np.clip(c['altitude_gain_s2']*(c['target_altitude_m']-h)-c['vertical_damping_s']*vr,-12.,12.)
            p=self.vehicle.properties(self._props,self.attached,self.fairing)[0]
            thrust=self.c['vehicle']['stages'][1]['thrust_N']
            sinp=np.clip((MU/np.linalg.norm(r)**2-vt*vt/np.linalg.norm(r)+desired_ar)/(thrust/p),-.7,.95)
            pitch=np.arcsin(sinp)
        target=np.cos(pitch)*horizontal+np.sin(pitch)*u
        if t>c.get('aoa_limit_start_s',25.) and qbar>1. and np.linalg.norm(vrel)>100 and np.dot(vrel,horizontal)>0:
            vh=unit(vrel); angle=np.arccos(np.clip(np.dot(target,vh),-1,1))
            # Continuous relaxation as atmosphere thins; avoids a command step at max-Q exit.
            limit=np.deg2rad(min(60.,c['aoa_command_limit_deg']*np.sqrt(max(1.,c['aoa_limit_min_q_Pa']/qbar))))
            if angle>limit:
                tangent=unit(target-vh*np.dot(target,vh))
                target=np.cos(limit)*vh+np.sin(limit)*tangent
        return unit(target)

    def evaluate(self,t,y,record=False):
        c=self.c; r=y[:3]; v=y[3:6]; Q=matrix(y[6:10]); w=y[10:13]
        props=np.maximum(y[13:15],0); self._props=props
        m,cg,I=self.vehicle.properties(props,self.attached,self.fairing)
        h=np.linalg.norm(r)-RE; e,n,u=local_frame(r)
        temp,p,rho,sound,muair=self.atm.sample(h)
        wind=self.atm.wind(h,t,r); vr=v-np.cross(OMEGA,r)-wind
        vb=Q.T@vr; speed=np.linalg.norm(vr); mach=speed/sound
        alpha=np.arctan2(np.linalg.norm(vb[1:]),vb[0]) if speed>1e-5 else 0.
        qbar=.5*rho*speed**2
        powered=self.phase in [1,2] and not self.cutoff
        ca,cn,cpnose,off=self.table.sample(mach,alpha,powered)
        S=c['aero']['reference_area_m2']
        fn=-qbar*S*cn*unit(np.r_[0.,vb[1:]]) if np.linalg.norm(vb[1:])>1e-10 else np.zeros(3)
        Fa=np.r_[-qbar*S*ca,0.,0.]+fn
        # The raw table describes the complete vehicle. Upper-stage reuse is explicit.
        nose=sum(s['length_m'] for s in c['vehicle']['stages'])
        if not self.fairing: nose-=c['vehicle']['fairing_length_m']
        cp=nose-cpnose+c['aero'].get('cp_station_offset_m',0.)
        Ma=np.cross(np.array([cp-cg,0.,0.]),Fa)
        idx=0 if self.phase==1 else 1; st=c['vehicle']['stages'][idx]
        engine=self.vehicle.offset[idx]+st['engine_pivot_m']
        arm=cg-engine
        Tfull=st['thrust_N']
        # Optional known nozzle-area correction. Default zero because source lacks area.
        Tfull=max(0.,Tfull+(st['reference_pressure_Pa']-p)*st['exit_area_m2'])
        throttle=0.
        if powered:
            schedule=np.interp(t,c['throttle']['time_s'],c['throttle']['fraction'])
            axial_cap=(m*G0*c['limits']['command_axial_g']-Fa[0])/max(Tfull,1.)
            throttle=np.clip(min(schedule,axial_cap),st['min_throttle'],1.)
        T=Tfull*throttle
        target=self.guidance(t,r,vr,v,h,qbar)
        tb=Q.T@target
        err=np.cross(np.array([1.,0.,0.]),tb)
        ctl=c['control']; wn=ctl['natural_frequency_rad_s']
        wref=Q.T@OMEGA
        angular_command=wn*wn*err-2*ctl['damping_ratio']*wn*(w-wref)
        anorm=np.linalg.norm(angular_command)
        amax=np.deg2rad(ctl.get('angular_acceleration_limit_deg_s2',.3))
        if self.phase==2: amax=np.deg2rad(ctl.get('upper_angular_acceleration_limit_deg_s2',1.5))
        if anorm>amax: angular_command*=amax/anorm
        # Axial roll is held by a separately stated RCS allocation, not engine gimbal.
        mdot=T/(st['isp_s']*G0) if powered else 0.
        dp=np.zeros(2); dp[idx]=-mdot
        Idot=np.zeros(3)
        cgdot=0.; cgddot=0.
        if mdot:
            altered=props.copy(); altered[idx]=max(0.,altered[idx]-1.)
            minus=self.vehicle.properties(altered,self.attached,self.fairing)
            Idot=(minus[2]-I)*mdot
            plusprops=props.copy(); plusprops[idx]+=1.
            plus=self.vehicle.properties(plusprops,self.attached,self.fairing)
            cgp=(plus[1]-minus[1])/2
            cgdot=-cgp*mdot
            # Slow mass-flow variation: exact schedule slope or late-burn g-limit slope.
            knot=min(len(c['throttle']['time_s'])-2,max(0,np.searchsorted(c['throttle']['time_s'],t)-1))
            slope=(c['throttle']['fraction'][knot+1]-c['throttle']['fraction'][knot])/(c['throttle']['time_s'][knot+1]-c['throttle']['time_s'][knot])
            mdotdot=(Tfull*slope/(st['isp_s']*G0) if schedule<axial_cap and st['min_throttle']<throttle<1 else
                     -mdot*c['limits']['command_axial_g']/st['isp_s'] if axial_cap<schedule and throttle>st['min_throttle'] else 0.)
            cgddot=(plus[1]-2*cg+minus[1])*mdot**2-cgp*mdotdot
        flux=mdot*arm*arm*np.r_[0.,w[1:]]
        Mrequired=I*angular_command+np.cross(w,I*w)+Idot*w+flux-Ma
        limit=np.deg2rad(st['gimbal_limit_deg']); rate=np.deg2rad(st['gimbal_rate_deg_s'])
        # State y[15:17] represents force-direction tilt in body y,z (radians).
        desired=np.array([-Mrequired[2],Mrequired[1]])/max(T*arm,1.) if T>0 and arm>0 else np.zeros(2)
        dn=np.linalg.norm(desired)
        if dn>np.sin(limit): desired*=np.sin(limit)/dn
        desired=np.arcsin(np.clip(desired,-1.,1.))
        gd=(desired-y[15:17])/ctl['actuator_time_constant_s']
        if np.linalg.norm(gd)>rate: gd*=rate/np.linalg.norm(gd)
        gvec=y[15:17].copy(); gn=np.linalg.norm(gvec)
        if gn>limit: gvec*=limit/gn
        Ft=T*np.r_[np.sqrt(max(0.,1.-np.sum(np.sin(gvec)**2))),np.sin(gvec)]
        Mt=np.cross(np.array([-arm,0.,0.]),Ft)
        rcscap=ctl['rcs_moment_Nm'] if not powered else ctl['roll_rcs_moment_Nm']
        Mr=np.zeros(3)
        if not powered:
            mrnorm=np.linalg.norm(Mrequired)
            Mr=Mrequired*min(1.,rcscap/max(mrnorm,1e-12))
        else: Mr[0]=np.clip(Mrequired[0],-rcscap,rcscap)
        wdot=(Ma+Mt+Mr-np.cross(w,I*w)-Idot*w-flux)/I
        gravity=-MU*r/np.linalg.norm(r)**3
        acceleration=Q@(Fa+Ft)/m+gravity
        point=np.array([c['vehicle']['payload_station_m']-cg,0.,0.])
        proper=(Fa+Ft)/m+np.cross(wdot,point)+np.cross(w,np.cross(w,point))
        proper+=2*np.cross(w,np.array([-cgdot,0.,0.]))+np.array([-cgddot,0.,0.])
        vh=unit(v)
        steering=T/m*(1.-np.clip(np.dot(Q@unit(Ft),vh),-1,1)) if T else 0.
        gimbal_loss=T/m*(1.-Ft[0]/T) if T else 0.
        drag_loss=-np.dot(Q@Fa,vh)/m
        grav_loss=-np.dot(gravity,vh)
        deriv=np.r_[v,acceleration,quaternion_derivative(y[6:10],w),wdot,dp,gd,
                    T/m,steering,gimbal_loss,drag_loss,grav_loss]
        if not record: return deriv
        available=T*max(arm,0.)*np.sin(limit)+(ctl['rcs_moment_Nm'] if not powered else 0.)
        aero_moment=np.linalg.norm(Ma[1:]); demand=np.linalg.norm(Mrequired[1:])
        relative=v-np.cross(OMEGA,r)
        def elevation(a): return np.rad2deg(np.arctan2(np.dot(a,u),np.linalg.norm(a-np.dot(a,u)*u)))
        inertial_speed=np.linalg.norm(v); radial=np.dot(v,u)
        hv=np.cross(r,v); ecc=np.linalg.norm(np.cross(v,hv)/MU-unit(r))
        energy=.5*inertial_speed**2-MU/np.linalg.norm(r)
        sma=-MU/(2*energy) if energy<0 else np.inf
        out=dict(t_s=t,phase=self.phase,altitude_m=h,mass_kg=m,cg_from_base_m=cg,
                 Ixx_kg_m2=I[0],Iyy_kg_m2=I[1],Izz_kg_m2=I[2],prop1_kg=props[0],prop2_kg=props[1],
                 cg_rate_m_s=cgdot,cg_acceleration_m_s2=cgddot,
                 airspeed_m_s=speed,inertial_speed_m_s=inertial_speed,vertical_speed_m_s=radial,
                 mach=mach,temperature_K=temp,pressure_Pa=p,density_kg_m3=rho,wind_m_s=np.linalg.norm(wind),
                 reynolds_length=rho*speed*(sum(s['length_m'] for s in c['vehicle']['stages'])-(0. if self.attached else c['vehicle']['stages'][0]['length_m'])-(0. if self.fairing else c['vehicle']['fairing_length_m']))/muair,
                 q_Pa=qbar,alpha_deg=np.rad2deg(alpha),qalpha_Pa_deg=qbar*np.rad2deg(alpha),
                 pitch_deg=elevation(Q[:,0]),heading_deg=np.rad2deg(np.arctan2(np.dot(Q[:,0],e),np.dot(Q[:,0],n)))%360,
                 flight_path_deg=elevation(relative),inertial_flight_path_deg=elevation(v),command_pitch_deg=elevation(target),
                 throttle=throttle,thrust_N=T,gimbal_y_deg=np.rad2deg(gvec[0]),gimbal_z_deg=np.rad2deg(gvec[1]),
                 gimbal_total_deg=np.rad2deg(np.linalg.norm(gvec)),gimbal_rate_deg_s=np.rad2deg(np.linalg.norm(gd)),
                 axial_g=proper[0]/G0,lateral_g=np.linalg.norm(proper[1:])/G0,cm_proper_g=np.linalg.norm(Fa+Ft)/m/G0,
                 aero_moment_Nm=aero_moment,available_moment_Nm=available,required_moment_Nm=demand,
                 actual_control_moment_Nm=np.linalg.norm((Mt+Mr)[1:]),moment_margin_Nm=available-aero_moment,
                 ratio2_margin_Nm=available-c['limits']['control_ratio']*aero_moment,
                 normal_force_N=np.linalg.norm(fn),axial_aero_force_N=-Fa[0],cp_from_base_m=cp,
                 aero_out_of_domain=int(off),aero_configuration_proxy=int(not self.attached or not self.fairing),
                 ideal_dv_m_s=y[17],steering_loss_m_s=y[18],gimbal_loss_m_s=y[19],drag_loss_m_s=y[20],gravity_loss_m_s=y[21],
                 osculating_perigee_m=sma*(1-ecc)-RE,osculating_apogee_m=sma*(1+ecc)-RE,
                 quaternion_norm=np.linalg.norm(y[6:10]))
        for i,k in enumerate(['x','y','z']): out[f'eci_{k}_m']=r[i]; out[f'eci_v{k}_m_s']=v[i]; out[f'body_rate_{k}_deg_s']=np.rad2deg(w[i])
        for i,k in enumerate(['x','y','z','w']): out[f'quaternion_{k}']=y[6+i]
        return out

    def integrate(self,y,t0,t1,events=None):
        c=self.c['solver']
        if c.get('verbose',False): print(f'Integrating phase {self.phase}: t={t0:.2f} s',flush=True)
        atol=np.r_[np.full(3,c['position_atol_m']),np.full(3,c['velocity_atol_m_s']),
                   np.full(4,1e-9),np.full(3,1e-9),np.full(2,1e-4),np.full(2,1e-9),np.full(5,1e-5)]
        sol=solve_ivp(self.evaluate,(t0,t1),y,method='RK45',rtol=c['rtol'],atol=atol,
                      max_step=c['max_step_s'],dense_output=True,events=events)
        if not sol.success: raise RuntimeError(sol.message)
        if c.get('verbose',False): print(f'  reached {sol.t[-1]:.2f} s, {sol.nfev} RHS evaluations',flush=True)
        ts=np.arange(t0,sol.t[-1],c['output_step_s'])
        ts=np.unique(np.r_[ts,sol.t[-1]])
        self.rows.extend(self.evaluate(float(t),sol.sol(t),True) for t in ts)
        return sol.y[:,-1],float(sol.t[-1]),sol

    def run(self):
        y=self.initial(); t=0.; c=self.c
        def burnout1(t,y): return y[13]-c['vehicle']['stages'][0]['residual_kg']
        burnout1.terminal=True; burnout1.direction=-1
        def ground(t,y): return np.linalg.norm(y[:3])-RE+1.
        ground.terminal=True; ground.direction=-1
        self.events.append({'event':'liftoff','t_s':0.})
        y,t,sol=self.integrate(y,t,c['solver']['max_time_s'],[burnout1,ground])
        if not len(sol.t_events[0]):
            self.events.append({'event':'terminated_before_stage1_burnout','t_s':t}); return self.rows,self.events
        self.phase=0
        self.events.append({'event':'MECO','t_s':t})
        y,t,coastsol=self.integrate(y,t,min(t+c['separation']['delay_s'],c['solver']['max_time_s']),[ground])
        if len(coastsol.t_events[0]) or t>=c['solver']['max_time_s']:
            self.events.append({'event':'ground_contact' if len(coastsol.t_events[0]) else 'time_limit','t_s':t})
            return self.rows,self.events
        oldm,oldcg,_=self.vehicle.properties(y[13:15],True,True)
        self.attached=False
        newm,newcg,_=self.vehicle.properties(y[13:15],False,True)
        shift=matrix(y[6:10])@np.array([newcg-oldcg,0.,0.])
        y[:3]+=shift; y[3:6]+=np.cross(matrix(y[6:10])@y[10:13],shift)
        dv=c['separation']['upper_delta_v_m_s']; y[3:6]+=matrix(y[6:10])[:,0]*dv
        self.events.append({'event':'stage_separation','t_s':t,'shed_mass_kg':oldm-newm,'cg_shift_m':newcg-oldcg,
                            'upper_delta_v_m_s':dv,'lower_delta_v_m_s':-newm/(oldm-newm)*dv})
        self.phase=2; self.ignition_time=t
        cg_before=self.vehicle.properties(y[13:15],False,self.fairing)[1]
        y[14]-=c['vehicle']['stages'][1]['startup_propellant_kg']
        cg_after=self.vehicle.properties(y[13:15],False,self.fairing)[1]
        shift=matrix(y[6:10])@np.array([cg_after-cg_before,0.,0.])
        y[:3]+=shift; y[3:6]+=np.cross(matrix(y[6:10])@y[10:13],shift)
        self.events.append({'event':'stage2_ignition','t_s':t})
        def burnout2(t,y): return y[14]-c['vehicle']['stages'][1]['residual_kg']
        burnout2.terminal=True; burnout2.direction=-1
        def insertion(t,y):
            r=y[:3]; v=y[3:6]; radial=np.dot(v,unit(r)); vt=np.linalg.norm(v-radial*unit(r))
            return vt-np.sqrt(MU/np.linalg.norm(r))
        insertion.terminal=True; insertion.direction=1
        def fairing(t,y): return np.linalg.norm(y[:3])-RE-c['separation']['fairing_altitude_m']
        fairing.terminal=True; fairing.direction=1
        event='time_limit'
        while t<c['solver']['max_time_s']:
            if self.fairing and np.linalg.norm(y[:3])-RE >= c['separation']['fairing_altitude_m']:
                before=self.vehicle.properties(y[13:15],False,True)
                self.fairing=False
                after=self.vehicle.properties(y[13:15],False,False)
                shift=matrix(y[6:10])@np.array([after[1]-before[1],0.,0.])
                y[:3]+=shift; y[3:6]+=np.cross(matrix(y[6:10])@y[10:13],shift)
                self.events.append({'event':'fairing_jettison','t_s':t,'shed_mass_kg':before[0]-after[0]})
            es=[burnout2,insertion,ground]+([fairing] if self.fairing else [])
            y,t,sol=self.integrate(y,t,c['solver']['max_time_s'],es)
            if len(sol.t_events[0]): event='stage2_propellant_limit'; break
            if len(sol.t_events[1]): event='orbital_speed_cutoff'; break
            if len(sol.t_events[2]): event='ground_contact'; break
            if self.fairing and len(sol.t_events[3]):
                # Move across root by a negligible distance to execute mass jump next loop.
                y[:3]+=unit(y[:3])*1e-6
                continue
            event='time_limit'; break
        self.events.append({'event':event,'t_s':t})
        if event=='orbital_speed_cutoff':
            self.phase=3; self.cutoff=True
            y,t,_=self.integrate(y,t,t+c['guidance']['post_cutoff_coast_s'],[ground])
            self.events.append({'event':'end_coast','t_s':t})
        return self.rows,self.events


def summarize(df,events,c):
    """Return the canonical engineering summary and sampled governing load cases."""
    peaks={}
    for key in ['q_Pa','qalpha_Pa_deg','axial_g','lateral_g','normal_force_N','axial_aero_force_N','aero_moment_Nm','required_moment_Nm','thrust_N','gimbal_total_deg']:
        row=df.loc[df[key].abs().idxmax()]
        peaks[key]={k:float(row[k]) for k in ['t_s',key,'altitude_m','mach','alpha_deg','mass_kg','aero_out_of_domain','aero_configuration_proxy']}
    powered=df[df.thrust_N>0]
    loaded=df[df.q_Pa>100]
    active=df[df.aero_moment_Nm>1.]
    coast=df[df.thrust_N==0]
    ratios=active.available_moment_Nm/active.aero_moment_Nm
    powered_ratios=powered[powered.aero_moment_Nm>1.].available_moment_Nm/powered[powered.aero_moment_Nm>1.].aero_moment_Nm
    cutoff=next((x['t_s'] for x in events if x['event']=='orbital_speed_cutoff'),None)
    end=df.iloc[-1] if cutoff is None else df.iloc[np.argmin(np.abs(df.t_s-cutoff))]
    orbit_ok=bool(cutoff is not None and abs(end.altitude_m-c['guidance']['target_altitude_m'])<5000 and
                  abs(end.vertical_speed_m_s)<30 and end.osculating_perigee_m>190000)
    stage2=c['vehicle']['stages'][1]
    burnout_mass=stage2['dry_mass_kg']+c['vehicle']['payload_kg']+stage2['residual_kg']
    reserve=stage2['isp_s']*G0*np.log((stage2['dry_mass_kg']+c['vehicle']['payload_kg']+end.prop2_kg)/burnout_mass)
    checks=dict(axial_acceleration_pass=bool(df.axial_g.abs().max()<=c['limits']['axial_g']),
                lateral_acceleration_pass=bool(df.lateral_g.max()<=c['limits']['lateral_g']),
                nominal_moment_inequality_all_samples=bool((df.moment_margin_Nm>=-1e-8).all()),
                nominal_moment_ratio_requirement_all_samples=bool((df.ratio2_margin_Nm>=-1e-8).all()),
                moment_demand_within_capacity_powered=bool((powered.required_moment_Nm<=powered.available_moment_Nm+1e-6).all()),
                circular_200km_target_screen_pass=orbit_ok,
                validated_control_authority=False)
    # Rigid-body load cases: interface proper acceleration times supported payload.
    cases=[]
    for name,key in [('Max-Q','q_Pa'),('Max-Qalpha','qalpha_Pa_deg'),('Max axial acceleration','axial_g'),
                     ('Max lateral acceleration','lateral_g'),('Max aerodynamic moment','aero_moment_Nm'),('Max TVC demand','required_moment_Nm'),
                     ('Max thrust','thrust_N'),('Max aerodynamic normal force','normal_force_N'),('Max aerodynamic axial force','axial_aero_force_N')]:
        row=df.loc[df[key].abs().idxmax()].to_dict()
        row['case']=name
        row['payload_axial_force_N']=row['axial_g']*G0*c['vehicle']['payload_kg']
        row['payload_lateral_force_N']=row['lateral_g']*G0*c['vehicle']['payload_kg']
        cases.append(row)
    checks['significant_q_table_domain_pass']=bool((loaded.aero_out_of_domain==0).all())
    checks['powered_moment_ratio_requirement_pass']=bool((powered.ratio2_margin_Nm>=-1e-8).all())
    summary=dict(peaks=peaks,checks=checks,events=events,
        duration_s=float(df.t_s.max()),initial_mass_kg=float(df.mass_kg.iloc[0]),
        ideal_remaining_delta_v_m_s=float(reserve),reference_TMI_delta_v_m_s=c.get('mission',{}).get('TMI_delta_v_m_s',3570.5),
        min_moment_ratio_above_1Nm=float(ratios.min()) if len(ratios) else None,
        min_powered_moment_ratio_above_1Nm=float(powered_ratios.min()) if len(powered_ratios) else None,
        worst_ratio2_margin_Nm=float(df.ratio2_margin_Nm.min()),
        coast_peak_aero_moment_Nm=float(coast.aero_moment_Nm.max()) if len(coast) else 0.,
        additional_coast_moment_for_ratio2_Nm=float(np.maximum(0,coast.aero_moment_Nm*c['limits']['control_ratio']-coast.available_moment_Nm).max()) if len(coast) else 0.,
        aero_outside_table_samples=int(df.aero_out_of_domain.sum()),
        aero_outside_table_above_100Pa_samples=int(loaded.aero_out_of_domain.sum()),
        post_separation_aero_proxy_samples=int(df.aero_configuration_proxy.sum()),
        separation_average_acceleration_g=c['separation']['upper_delta_v_m_s']/c['separation']['impulse_duration_s']/G0,
        peak_aero_outside_table_q_Pa=float(df.loc[df.aero_out_of_domain==1,'q_Pa'].max()) if df.aero_out_of_domain.any() else 0.,
        cutoff_state={k:float(end[k]) for k in ['t_s','altitude_m','inertial_speed_m_s','vertical_speed_m_s',
                     'osculating_perigee_m','osculating_apogee_m','prop2_kg','ideal_dv_m_s','steering_loss_m_s',
                     'gimbal_loss_m_s','drag_loss_m_s','gravity_loss_m_s']},
        assumptions=c['assumptions'])
    return summary,pd.DataFrame(cases)

def run_trajectory(config=None):
    """Run the shared model and return ``(trajectory, summary, simulation)``.

    ``config`` may be an editable dictionary or a JSON path. Omitting it loads
    the selected low-AoA case from ``config/trajectory.json``. Output samples
    include both sides of instantaneous phase/mass changes, so event times may
    repeat intentionally; do not silently discard those rows for load checks.
    Units appear in column names; ``mach`` and quaternions are dimensionless.
    ``simulation.load_cases`` contains sampled maximum-load rows.

    Numerical verification is distinct from engineering verification: the
    returned summary retains failed coast-control and aero-coverage checks.
    """
    if config is None or isinstance(config, (str, Path)):
        config = load_config(config)
    simulation = Simulation(config)
    rows, events = simulation.run()
    trajectory = pd.DataFrame(rows)
    summary, cases = summarize(trajectory, events, simulation.c)
    simulation.load_cases = cases
    return trajectory, summary, simulation

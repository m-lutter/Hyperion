"""Annotated plots shared by Hyperion notebooks."""
from pathlib import Path
import numpy as np
from .trajectory import RE

def plot_results(df,s,c,out=None):
    """Return five annotated figure sheets; optionally save PNGs to ``out``.

    The caller owns the returned figures and may display or close them. This
    helper neither selects a backend nor modifies the input trajectory frame.
    """
    import matplotlib.pyplot as plt
    figures=[]
    if out is not None:
        out=Path(out)
        out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
                         'grid.alpha':.18,'figure.facecolor':'#f7f9fc','axes.facecolor':'white','lines.linewidth':1.7})
    times=[s['peaks'][k]['t_s'] for k in ['q_Pa','qalpha_Pa_deg','axial_g','lateral_g']]
    colors=['#b76404','#b8226b','#bd2424','#7a48aa']
    labels=['Max-Q','Max-Qα','Max axial g','Max lateral g']
    def marks(ax,allmarks=True):
        for x,col,lab in zip(times,colors,labels):
            if allmarks: ax.axvline(x,color=col,ls=':',lw=1,alpha=.8,label=lab)
        for ev in s['events']:
            if ev['event'] in ['stage_separation','fairing_jettison','orbital_speed_cutoff']:
                ax.axvline(ev['t_s'],color='#546780',ls='--',lw=.8)
        ax.set_xlabel('Time since release (s)')
    def save(fig,name,title):
        fig.suptitle(title,fontsize=16,fontweight='bold',x=.05,ha='left')
        fig.text(.05,.015,'Hyperion • Current configuration • Preliminary rigid-body model; actuator capability assumed',fontsize=9,color='#586579')
        if name!='04_orientation_atmosphere.png':
            from matplotlib.lines import Line2D
            fig.legend([Line2D([0],[0],color=co,ls=':') for co in colors],labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,.952),frameon=False)
        fig.tight_layout(rect=[0,.035,1,.915])
        if out is not None: fig.savefig(out/name,dpi=150)
        figures.append(fig)
    t=df.t_s
    fig,ax=plt.subplots(3,2,figsize=(14,11))
    ax[0,0].plot(t,df.altitude_m/1000);ax[0,0].axhline(c['guidance']['target_altitude_m']/1000,color='gray',ls='--');ax[0,0].set_ylabel('Altitude (km)')
    # ECEF ground-track arc from initial position, independent of Earth's rotation.
    theta=7.292115e-5*t.to_numpy();x=df.eci_x_m.to_numpy();y=df.eci_y_m.to_numpy();z=df.eci_z_m.to_numpy()
    re=np.column_stack([np.cos(theta)*x+np.sin(theta)*y,-np.sin(theta)*x+np.cos(theta)*y,z]);re/=np.linalg.norm(re,axis=1)[:,None]
    down=RE*np.arccos(np.clip(re@re[0],-1,1))/1000
    ax[0,1].plot(down,df.altitude_m/1000);ax[0,1].set(xlabel='Ground arc from launch (km)',ylabel='Altitude (km)')
    for tt,col,lab in zip(times,colors,labels):
        j=np.argmin(abs(t-tt));ax[0,1].scatter(down[j],df.altitude_m.iloc[j]/1000,color=col,s=30,label=lab)
    ax[1,0].plot(t,df.pitch_deg,label='Body elevation');ax[1,0].plot(t,df.command_pitch_deg,ls='--',label='Command');ax[1,0].set_ylabel('Orientation / elevation (deg)');ax[1,0].legend()
    ax[1,1].plot(t,df.flight_path_deg,label='Earth-relative');ax[1,1].plot(t,df.inertial_flight_path_deg,label='Inertial',alpha=.6);ax[1,1].set_ylabel('Flight path angle (deg)');ax[1,1].legend()
    ax[2,0].plot(t,df.alpha_deg);ax[2,0].axhline(4,color='red',ls='--',label='Table AoA boundary');ax[2,0].set_ylabel('Total angle of attack (deg)');ax[2,0].legend()
    ax[2,1].plot(t,df.thrust_N/1e6,label='Thrust');ax[2,1].set_ylabel('Commanded thrust (MN)')
    for axis in ax.flat:
        if axis is not ax[0,1]: marks(axis)
    ax[0,1].legend(fontsize=8)
    save(fig,'01_trajectory.png','Trajectory, attitude, and thrust')
    fig,ax=plt.subplots(3,2,figsize=(14,11))
    ax[0,0].plot(t,df.q_Pa/1000);ax[0,0].set_ylabel('Dynamic pressure (kPa)')
    ax[0,1].plot(t,df.qalpha_Pa_deg/1000);ax[0,1].set_ylabel('q × |α| (kPa·deg)')
    if s['peaks']['qalpha_Pa_deg']['aero_out_of_domain']:
        ax[0,1].text(.43,.80,'Max-Qα is outside the 4° AoA table.\nForce/moment prediction is conditional.',transform=ax[0,1].transAxes,color='#b8226b',fontsize=9)
    ax[1,0].plot(t,df.axial_g,label='Payload axial');ax[1,0].axhline(c['limits']['axial_g'],color='red',ls='--',label=f"{c['limits']['axial_g']:g} g limit");ax[1,0].set_ylabel('Axial proper acceleration (g)');ax[1,0].legend()
    ax[1,1].plot(t,df.lateral_g);ax[1,1].axhline(c['limits']['lateral_g'],color='red',ls='--',label=f"{c['limits']['lateral_g']:g} g limit");ax[1,1].set_ylabel('Payload lateral acceleration (g)');ax[1,1].legend()
    ax[2,0].plot(t,df.normal_force_N/1000,label='Normal');ax[2,0].plot(t,df.axial_aero_force_N/1000,label='Axial');ax[2,0].set_ylabel('Aerodynamic force (kN)');ax[2,0].legend()
    for fk,offset in [('normal_force_N',(15,15)),('axial_aero_force_N',(15,-20))]:
        pk=s['peaks'][fk]
        ax[2,0].scatter(pk['t_s'],pk[fk]/1000,s=25,color='#b8226b')
        ax[2,0].annotate(f"{pk[fk]/1000:.2f} kN @ {pk['t_s']:.1f} s",(pk['t_s'],pk[fk]/1000),xytext=offset,textcoords='offset points',fontsize=8)
    ax[2,1].plot(t,df.aero_moment_Nm/1e6);ax[2,1].set_ylabel('Aerodynamic moment about CM (MN·m)')
    for axis,key,scale,units in [(ax[0,0],'q_Pa',1000,'kPa'),(ax[0,1],'qalpha_Pa_deg',1000,'kPa·deg'),
                               (ax[1,0],'axial_g',1,'g'),(ax[1,1],'lateral_g',1,'g'),(ax[2,1],'aero_moment_Nm',1e6,'MN·m')]:
        peak=s['peaks'][key];axis.scatter(peak['t_s'],peak[key]/scale,s=35,color='#b8226b',zorder=5)
        axis.annotate(f"{peak[key]/scale:.3f} {units} @ {peak['t_s']:.1f} s",(peak['t_s'],peak[key]/scale),xytext=(12,18 if key=='lateral_g' else -22),textcoords='offset points',fontsize=9)
    for axis in ax.flat:marks(axis)
    ax[0,0].legend(loc='upper right',fontsize=8)
    save(fig,'02_load_cases.png','Maximum aerodynamic and payload loads')
    fig,ax=plt.subplots(3,2,figsize=(14,11))
    ax[0,0].plot(t,df.gimbal_y_deg,label='Body y tilt');ax[0,0].plot(t,df.gimbal_z_deg,label='Body z tilt');ax[0,0].plot(t,df.gimbal_total_deg,label='Total',color='black',alpha=.5);ax[0,0].set_ylabel('Gimbal angle (deg)');ax[0,0].legend()
    ax[0,1].plot(t,df.throttle*100);ax[0,1].set_ylabel('Throttle command (%)')
    for name,lab in [('aero_moment_Nm','Aero disturbance'),('available_moment_Nm','Available control'),('required_moment_Nm','Tracking demand')]:
        ax[1,0].plot(t,df[name]/1e6,label=lab)
    ax[1,0].set_ylabel('Moment magnitude (MN·m)');ax[1,0].legend(fontsize=8)
    ax[1,1].plot(t,df.ratio2_margin_Nm/1e6);ax[1,1].axhline(0,color='red',ls='--');ax[1,1].fill_between(t,df.ratio2_margin_Nm/1e6,0,where=df.ratio2_margin_Nm<0,color='red',alpha=.3);ax[1,1].set_ylabel('Available − 2 × aero (MN·m)')
    meco=next((ev['t_s'] for ev in s['events'] if ev['event']=='MECO'),None)
    if meco is not None:
        ins=ax[1,1].inset_axes([.48,.30,.48,.42])
        ins.plot(t,df.ratio2_margin_Nm/1000,color='#b8226b');ins.axhline(0,color='red',ls='--',lw=.8)
        ins.set_xlim(meco-.2,meco+c['separation']['delay_s']+.2)
        deficit=s['additional_coast_moment_for_ratio2_Nm']/1000
        inset_scale=max(1.,deficit)
        ins.set_ylim(-inset_scale*1.25,inset_scale*.35);ins.tick_params(labelsize=7)
        ins.set_title('Engine-off gap (kN·m)',fontsize=8)
        ax[1,1].text(.48,.86,f'Coast requirement shortfall: {deficit:.1f} kN·m',transform=ax[1,1].transAxes,color='#b8226b',fontsize=9)
    ax[2,0].plot(t,df.mass_kg/1000,label='Mass');ax[2,0].set_ylabel('Mass (tonnes)');other=ax[2,0].twinx();other.plot(t,df.cg_from_base_m,color='#b76404');other.set_ylabel('CM from original base (m)',color='#b76404')
    for key,lab in [('steering_loss_m_s','Thrust direction'),('gimbal_loss_m_s','Gimbal diagnostic (non-additive)'),('drag_loss_m_s','Aero work'),('gravity_loss_m_s','Gravity')]:ax[2,1].plot(t,df[key],label=lab)
    ax[2,1].set_ylabel('Integrated velocity losses (m/s)');ax[2,1].legend(fontsize=8)
    for axis in ax.flat:marks(axis)
    ax[0,1].legend(loc='lower left',fontsize=8)
    save(fig,'03_control_and_mass.png','Control authority, gimbal motion, and mass depletion')
    fig,ax=plt.subplots(3,2,figsize=(14,11))
    for k in ['x','y','z','w']:ax[0,0].plot(t,df['quaternion_'+k],label=k)
    ax[0,0].set_ylabel('Body-to-ECI quaternion');ax[0,0].legend(ncol=4)
    for k in ['x','y','z']:ax[0,1].plot(t,df['body_rate_'+k+'_deg_s'],label=k)
    ax[0,1].set_ylabel('Body angular rates (deg/s)');ax[0,1].legend()
    ax[1,0].plot(t,df.temperature_K);ax[1,0].set_ylabel('Temperature (K)')
    ax[1,1].semilogy(t,np.maximum(df.pressure_Pa,1e-12),label='Pressure (Pa)');ax[1,1].semilogy(t,np.maximum(df.density_kg_m3,1e-12),label='Density (kg/m³)');ax[1,1].legend()
    ax[2,0].plot(t,df.wind_m_s);ax[2,0].set_ylabel('Wind speed (m/s)')
    ax[2,1].plot(t,df.Iyy_kg_m2/1e6,label='Pitch/yaw');ax[2,1].plot(t,df.Ixx_kg_m2/1e6,label='Roll');ax[2,1].set_ylabel('Principal inertia (10⁶ kg·m²)');ax[2,1].legend()
    for axis in ax.flat:marks(axis,False)
    save(fig,'04_orientation_atmosphere.png','Full attitude state and atmospheric environment')
    fig,ax=plt.subplots(2,1,figsize=(14,9),sharex=True)
    ax[0].plot(t,df.mach,color='#205e96',label='Air-relative Mach number')
    ax[0].axhline(1,color='#8b98a7',ls='--',lw=.9,label='Mach 1')
    ax[0].set_ylabel('Mach number (air-relative speed / local sound speed)')
    ax[0].legend(loc='upper left',fontsize=9)
    ax[1].plot(t,df.airspeed_m_s/1000,color='#205e96',label='Air-relative speed')
    ax[1].plot(t,df.inertial_speed_m_s/1000,color='#148579',ls='--',label='Earth-centered inertial speed')
    ax[1].set_ylabel('Speed (km/s)')
    ax[1].legend(loc='upper left',fontsize=9)
    for axis in ax:marks(axis)
    for event in s['events']:
        if event['event'] in ['stage_separation','fairing_jettison','orbital_speed_cutoff']:
            event_label={'stage_separation':'Stage separation','fairing_jettison':'Fairing jettison',
                         'orbital_speed_cutoff':'Engine cutoff'}[event['event']]
            ax[0].text(event['t_s']+3,.97,event_label,transform=ax[0].get_xaxis_transform(),
                       rotation=90,ha='left',va='top',fontsize=8,color='#546780')
    save(fig,'05_velocity_mach.png','Velocity and Mach history')
    return figures

"""Animate a clean climbing single roll and positive-North return curve."""
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from src.f16sim.air_data import air_data_from_body_velocity
from src.f16sim.attitude import euler_to_quaternion, quaternion_to_dcm
from src.f16sim.controllers import pitch_attitude_control
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16_feedback
from src.f16sim.trim import trim_straight_level
from src.f16sim.visualization import create_flight_animation, select_animation_frames

TRUE_AIRSPEED=502.0*FT_TO_METER; INITIAL_ALTITUDE=15_000.0*FT_TO_METER
CG_FRACTION=0.30; DT=0.01; DURATION=80.0
KQ,KTHETA,KP,KPHI,KR=5.0,0.5,5.0,1.0,50.0
ENTRY_END=2.5; CLIMB_ESTABLISHED=8.0; ROLL_START=10.0
ROLL_END_GUESS=23.5; POST_ROLL_END=31.0
TURN_ENTRY_END=36.0; TURN_ROLLOUT_START=65.0; TURN_ROLLOUT_END=71.0
CLIMB_PITCH=np.deg2rad(35.0); ROLL_GAMMA=np.deg2rad(9.5)
ROLL_RATE=np.deg2rad(75.0); TURN_BANK=np.deg2rad(-55.0); TURN_PITCH=np.deg2rad(2.0)
DEFAULT_FPS=30.0; DEFAULT_PLAYBACK_SPEED=6.0; AIRCRAFT_SCALE=150.0
CAMERA_VIEW_SIZE=10_000.0; CAMERA_VERTICAL_SIZE=5_000.0; CAMERA_SMOOTHING_TIME=4.0

def _smooth_step(t,a,b):
    if t<=a:return 0.0
    if t>=b:return 1.0
    x=(t-a)/(b-a); return 0.5-0.5*np.cos(np.pi*x)

def _commands(t,roll_end):
    climb=_smooth_step(t,ENTRY_END,5.5)*(1-_smooth_step(t,POST_ROLL_END,TURN_ENTRY_END))
    turn=_smooth_step(t,POST_ROLL_END,TURN_ENTRY_END)*(1-_smooth_step(t,TURN_ROLLOUT_START,TURN_ROLLOUT_END))
    pitch=CLIMB_PITCH*climb+TURN_PITCH*turn; bank=TURN_BANK*turn
    roll_rate=ROLL_RATE*_smooth_step(t,ROLL_START,ROLL_START+.75)*(1-_smooth_step(t,roll_end-1,roll_end))
    if t<ENTRY_END: phase="TRIMMED ENTRY"
    elif t<CLIMB_ESTABLISHED: phase="CLIMB ENTRY"
    elif t<ROLL_START: phase="ESTABLISHED CLIMB"
    elif t<roll_end: phase="SLOW SINGLE ROLL"
    elif t<POST_ROLL_END: phase="WINGS-LEVEL CLIMB"
    elif t<TURN_ENTRY_END: phase="POSITIVE-NORTH TURN ENTRY"
    elif t<TURN_ROLLOUT_START: phase="RETURN CURVE"
    elif t<TURN_ROLLOUT_END: phase="SMOOTH ROLLOUT"
    else: phase="STABLE RECOVERY"
    return pitch,bank,roll_rate,phase

def _euler(q):
    c=quaternion_to_dcm(q)
    return np.array([np.arctan2(c[1,2],c[2,2]),np.arcsin(np.clip(-c[0,2],-1,1)),np.arctan2(c[0,1],c[0,0])])

def _gamma(x):
    v=quaternion_to_dcm(x[6:10]).T@x[3:6]
    return -np.arctan2(v[2],np.linalg.norm(v[:2]))

def _simulate_once(trim,roll_end):
    trim_phi,trim_theta,_=_euler(trim["state"][6:10]); trim_p,trim_q,trim_r=trim["state"][10:13]
    def law(t,x):
        phi,theta,_=_euler(x[6:10]); pitch,bank,roll_rate,_=_commands(t,roll_end)
        if ROLL_START<=t<=POST_ROLL_END:
            q_cmd=3.0*(ROLL_GAMMA-_gamma(x))*np.cos(phi)
            elevator=KQ*((x[11]-trim_q)-q_cmd)
        else:
            _,elevator=pitch_attitude_control(trim_theta+pitch,theta,x[11]-trim_q,Kq=KQ,Ktheta=KTHETA)
        p_cmd=roll_rate if ROLL_START<=t<=roll_end else KPHI*(trim_phi+bank-phi)
        return np.array([trim["throttle"],trim["elevator_deg"]+elevator,KP*((x[10]-trim_p)-p_cmd),KR*(x[12]-trim_r)])
    initial=trim["state"].copy(); initial[6:10]=euler_to_quaternion(0,trim_theta,np.deg2rad(90))
    times,states=simulate_f16_feedback(initial,DURATION,DT,law,CG_FRACTION)
    controls=np.array([law(t,x) for t,x in zip(times,states)])
    rows=[_commands(t,roll_end) for t in times]
    commands=np.array([r[:2] for r in rows]); phases=np.array([r[3] for r in rows]); rates=np.empty(times.size)
    for i,(t,x) in enumerate(zip(times,states)):
        phi=_euler(x[6:10])[0]
        rates[i]=_commands(t,roll_end)[2] if ROLL_START<=t<=roll_end else KPHI*(trim_phi+_commands(t,roll_end)[1]-phi)
    return {"times":times,"states":states,"controls":controls,"trim":trim,"command_history":commands,"phase_history":phases,"rate_command_history":rates,"roll_end":roll_end}

def _integrated_roll(r):
    t=r["times"]; m=(t>=ROLL_START)&(t<=POST_ROLL_END)
    return np.rad2deg(np.trapezoid(r["states"][m,10],t[m]))

def simulate_maneuver():
    trim=trim_straight_level(TRUE_AIRSPEED,INITIAL_ALTITUDE,cg_fraction=CG_FRACTION)
    if not trim["success"]: raise RuntimeError(f"Unable to obtain trim: {trim['message']}")
    end=ROLL_END_GUESS
    for _ in range(10):
        result=_simulate_once(trim,end); total=_integrated_roll(result)
        if 352<=total<=368: break
        end=float(np.clip(end+(360-total)/35,18,28))
    result["integrated_roll"]=_integrated_roll(result); return result

def _roll_history(r):
    t=r["times"]; m=(t>=ROLL_START)&(t<=POST_ROLL_END); ids=np.flatnonzero(m); p=r["states"][m,10]
    cumulative=np.zeros(len(ids)); cumulative[1:]=np.rad2deg(np.cumsum(.5*(p[1:]+p[:-1])*np.diff(t[m])))
    return ids,cumulative

def print_diagnostics(r):
    t,x,u=r["times"],r["states"],r["controls"]; angles=np.array([_euler(q) for q in x[:,6:10]])
    heading=np.unwrap(angles[:,2]); gamma=np.rad2deg(np.array([_gamma(s) for s in x])); altitude=-x[:,2]/FT_TO_METER
    air=np.array([air_data_from_body_velocity(v) for v in x[:,3:6]]); body=np.rad2deg(x[:,10:13]); ids,cumulative=_roll_history(r); marks=[]
    print("Single-roll milestones")
    for target in (0.,90.,180.,270.,360.):
        local=int(np.argmin(abs(cumulative-target))); i=ids[local]; marks.append(i)
        print(f"{target:3.0f} deg: t={t[i]:.3f} s, N={x[i,0]/FT_TO_METER:.3f} ft, E={x[i,1]/FT_TO_METER:.3f} ft, alt={altitude[i]:.3f} ft, gamma={gamma[i]:.3f} deg, integrated={cumulative[local]:.3f} deg")
    start,finish=marks[0],marks[-1]; distance=np.linalg.norm(x[finish,:3]-x[start,:3])/FT_TO_METER
    print(f"Total integrated roll angle: {r['integrated_roll']:.3f} deg")
    print(f"Roll duration: {t[finish]-t[start]:.3f} s")
    print(f"Distance traveled during roll: {distance:.3f} ft")
    print(f"Altitude gain during roll: {altitude[finish]-altitude[start]:.3f} ft")
    print(f"Maximum |p|: {np.max(np.abs(body[(t>=ROLL_START)&(t<=POST_ROLL_END),0])):.3f} deg/s")
    print(f"completed full longitudinal rolls = {int(np.rint(abs(r['integrated_roll'])/360))}")
    turn=(t>=POST_ROLL_END)&(t<=TURN_ROLLOUT_END); pts=x[turn,:2]/FT_TO_METER; sample=pts[np.unique(np.linspace(0,len(pts)-1,101,dtype=int))]; radii=[]
    for a,b,c in zip(sample[:-2],sample[1:-1],sample[2:]):
        ab,bc,ac=np.linalg.norm(b-a),np.linalg.norm(c-b),np.linalg.norm(c-a); area2=abs((b-a)[0]*(c-a)[1]-(b-a)[1]*(c-a)[0])
        if area2>1e-8:radii.append(ab*bc*ac/(2*area2))
    ia=int(np.argmin(abs(t-POST_ROLL_END))); ib=int(np.argmin(abs(t-TURN_ROLLOUT_END))); north=x[:,0]/FT_TO_METER
    print("Return-curve diagnostics\nPreviously negative coordinate: North")
    print(f"Maximum positive North: {np.max(north):.3f} ft")
    print(f"Heading change: {np.rad2deg(heading[ib]-heading[ia]):.3f} deg")
    print(f"Maximum bank: {np.rad2deg(np.max(np.abs(angles[turn,0]))):.3f} deg")
    print(f"Return-curve radius estimate: {np.median(radii):.3f} ft")
    print(f"Final North / East: {north[-1]:.3f} / {x[-1,1]/FT_TO_METER:.3f} ft")
    print(f"VT range: {np.min(air[:,0])/FT_TO_METER:.3f} to {np.max(air[:,0])/FT_TO_METER:.3f} ft/s")
    print(f"Alpha range / maximum |beta|: {np.min(air[:,1]):.3f} to {np.max(air[:,1]):.3f} / {np.max(np.abs(air[:,2])):.3f} deg")
    print(f"Maximum elevator / aileron / rudder: {np.max(np.abs(u[:,1])):.3f} / {np.max(np.abs(u[:,2])):.3f} / {np.max(np.abs(u[:,3])):.3f} deg")
    print(f"Minimum altitude: {np.min(altitude):.3f} ft")
    print(f"Final phi / p,q,r: {np.rad2deg(angles[-1,0]):.3f} / {body[-1,0]:.3f}, {body[-1,1]:.3f}, {body[-1,2]:.3f} deg/s")
    failures=[]
    if not 350<=abs(r["integrated_roll"])<=370:failures.append("roll outside 350-370 deg")
    if int(np.rint(abs(r["integrated_roll"])/360))!=1:failures.append("roll count is not one")
    if np.min(air[:,1]) < -10 or np.max(air[:,1])>45 or np.max(abs(air[:,2]))>30:failures.append("aerodynamic range exceeded")
    if np.min(air[:,0])<=0 or np.min(altitude)<=0:failures.append("nonphysical speed or altitude")
    if np.max(abs(u[:,1:]))>25:failures.append("surface limit exceeded")
    if abs(np.rad2deg(angles[-1,0]))>3 or np.max(abs(body[-1]))>1:failures.append("final recovery not settled")
    if failures:raise RuntimeError("Acceptance checks failed: "+"; ".join(failures))
    print("Acceptance checks: PASSED")

def create_animation(r,fps=DEFAULT_FPS,playback_speed=DEFAULT_PLAYBACK_SPEED):
    animation=create_flight_animation(r["times"],r["states"],controls=r["controls"],command_history=r["command_history"],phase_history=r["phase_history"],rate_command_history=r["rate_command_history"],fps=fps,playback_speed=playback_speed,camera="chase",view_size=CAMERA_VIEW_SIZE,vertical_view_size=CAMERA_VERTICAL_SIZE,aircraft_scale=AIRCRAFT_SCALE,trail_duration=26,show_full_trajectory=True,look_ahead_distance=1500,camera_smoothing_time=CAMERA_SMOOTHING_TIME)
    fig,ax=animation._fig,animation._fig.axes[0]; fig.set_size_inches(15,8.5,forward=True); fig.subplots_adjust(left=.18,right=.72,bottom=.13,top=.82); ax.set_box_aspect((CAMERA_VIEW_SIZE,CAMERA_VIEW_SIZE,CAMERA_VERTICAL_SIZE)); ax.set_zlabel("Altitude MSL [ft]"); ax.set_title(""); fig.suptitle("F-16 Nonlinear 6DoF Climbing Single Roll and Return Curve",y=.96)
    left,right,control,phase,vertical=ax.texts[-5:]; left.set_position((-.27,.92)); right.set_position((1.18,.92)); control.set_position((1.18,.46)); phase.set_position((.5,1.035)); vertical.set_position((-.27,.04)); vertical.set_text("Vertical reference: mean sea level (0 ft)")
    for hud in (left,right,control):hud.set_fontsize(8.5)
    legend=ax.get_legend()
    if legend:legend.set_loc("lower center");legend.set_bbox_to_anchor((.5,-.16));legend.set_ncols(3)
    frames=select_animation_frames(r["times"],fps,playback_speed); original=animation._func
    def update(frame):
        artists=original(frame); i=int(frames[frame]); data=air_data_from_body_velocity(r["states"][i,3:6]); alt=-r["states"][i,2]/FT_TO_METER
        left.set_text("FLIGHT\n"+f"VT       {data[0]/FT_TO_METER:6.1f} ft/s\nAltitude {alt:6.0f} ft MSL\nalpha    {data[1]:6.1f} deg\nbeta     {data[2]:6.1f} deg"); ax.view_init(elev=25,azim=-52); return artists
    animation._func=update; return animation

def create_ground_track(r):
    n=(r["states"][:,0]-r["states"][0,0])/FT_TO_METER; e=(r["states"][:,1]-r["states"][0,1])/FT_TO_METER
    fig,ax=plt.subplots(figsize=(9,8));ax.plot(e,n,linewidth=2.2,label="Ground track");ax.scatter(e[0],n[0],color="C2",s=55,label="Start",zorder=3);ax.scatter(e[-1],n[-1],color="C3",marker="x",s=65,label="End",zorder=3);ax.set_xlabel("East displacement [ft]");ax.set_ylabel("North displacement [ft]");ax.set_title("Climbing Single Roll and Positive-North Return Curve");ax.axis("equal");ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();return fig

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--fps",type=float,default=DEFAULT_FPS);parser.add_argument("--speed",type=float,default=DEFAULT_PLAYBACK_SPEED);args=parser.parse_args()
    result=simulate_maneuver();print_diagnostics(result);create_ground_track(result);create_animation(result,args.fps,args.speed);plt.show()

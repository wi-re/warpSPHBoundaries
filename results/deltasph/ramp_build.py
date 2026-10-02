import sys; sys.path.insert(0,"python")
import math, numpy as np, torch, time
from edgebound.deltasph2d import DeltaSPH2D, DeltaSPHConfig, triangle_distance
from edgebound.dfsph2d import domain_scene
from edgebound.scene import Body, Scene, SurfaceRep
dev="cuda:0"; dp=0.02; g=9.81; L,Ht,Hw=1.6,1.0,0.5; c0=20*math.sqrt(g*Hw); bed=0.0
def build(angdeg=30, **kw):
    ang=math.radians(angdeg); x0=0.2; x1=1.6
    tri=np.array([[x0,bed],[x1,bed],[x1,bed+(x1-x0)*math.tan(ang)]])
    X,Y=np.meshgrid(dp*(np.arange(int(L/dp))+0.5), bed+dp*(np.arange(int(Hw/dp))+0.5), indexing="ij")
    pos=np.stack([X.ravel(),Y.ravel()],1); pos=pos[triangle_distance(pos,tri)>=0.5*dp]
    rho=1.0*(1.0+g*np.clip(bed+Hw-pos[:,1],0,None)/c0**2)
    dom=domain_scene("surface",(0,bed),(L,bed+Ht),4*dp,dev).bodies[0]
    sc=Scene([dom,Body(bodyId=1,reps=[SurfaceRep.polygon(tri,solid="inside")])],dev)
    return DeltaSPH2D(pos,np.zeros_like(pos),rho,dp,sc,DeltaSPHConfig(gravity=(0,-g),c0=c0,**kw),dev)

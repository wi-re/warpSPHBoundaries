"""Prescribed-state mixed-wall diagnostics using the native FluxBridge operators.

No pressure solve or timestep loop is performed. Compression is the supplied
boundary continuity contraction, not a porous pressure-density discretization.
"""
from itertools import permutations, product
from math import comb, factorial
import time
import numpy as np
from .compute import number
from ..fluxbridge import Geometry, Evaluator, Request, baseline
from ..fluxbridge.contract import monomials

DEFAULT_WALL = dict(dimension=2, geometry="wall", width=2., thickness=.4, depth=1.,
    fraction=.5, subdivisions=2, grading=1., angle=0., center=[0.,0.,0.],
    h=.6, kernel="w2", precision="float32", backend="numerical", device="cpu",preparation='host',
    resolution=9, slice=0., slice_plane='xy', degree=0, resistance=4., anisotropy=4., tensor_angle=0.,
    dt=.1, fluid=[0.,1.,0.], skeleton=[0.,0.,0.],
    fluid_gradient=[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],
    skeleton_gradient=[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],
    porous_enabled=True, exhaustive=False, segments=16, fibres=3, search='host')


def validate(payload):
    if not isinstance(payload,dict): raise ValueError("Configuration must be a JSON object")
    unknown=set(payload)-set(DEFAULT_WALL)-{'queries'}
    if unknown:raise ValueError('Unsupported wall configuration keys: '+', '.join(sorted(unknown)))
    c={**DEFAULT_WALL,**payload}
    for key,lo,hi,integer in (("dimension",2,3,True),("width",.2,6,False),
        ("thickness",.05,2,False),("depth",.1,4,False),("fraction",0,1,False),
        ("subdivisions",1,12,True),("grading",.5,3,False),("angle",-180,180,False),
        ("h",.05,3,False),("resolution",3,33,True),("slice",-4,4,False),
        ("degree",0,3,True),("resistance",0,1000,False),("anisotropy",1,100,False),
        ("tensor_angle",-180,180,False),("dt",0,10,False),("segments",8,64,True),
        ("fibres",1,8,True)):
        c[key]=number(c[key],key,lo,hi,integer)
    for key,choices in (("geometry",("wall","disk","fibres")),("kernel",("w2","w4","w6","cubic")),
        ("precision",("float32","float64")),("backend",("numerical","analytic")),("device",("cpu","cuda:0")),
        ("slice_plane",('xy','xz','yz')),("search",('host','device'))):
        if c[key] not in choices: raise ValueError(f"Unknown {key}")
    if c['dimension']==3 and c['geometry']!='wall':raise ValueError("3D explorer currently requires a tetrahedral wall")
    if c['backend']=='analytic' and c['precision']=='float32':
        raise ValueError("Analytic raw fp32 remains deferred; select numerical fp32 or analytic fp64")
    if c['preparation'] not in ('host','device') or (c['preparation']=='device' and (c['dimension']!=3 or c['backend']!='numerical')):
        raise ValueError('Device panel preparation requires the 3D numerical tetrahedral provider')
    for key in ('porous_enabled','exhaustive'):
        if not isinstance(c[key],bool):raise ValueError(f"{key} must be boolean")
    for key,shape in (("center",(3,)),("fluid",(3,)),("skeleton",(3,)),
        ("fluid_gradient",(3,3)),("skeleton_gradient",(3,3))):
        a=np.asarray(c[key],float)
        if a.shape!=shape or not np.isfinite(a).all() or np.max(abs(a))>100:
            raise ValueError(f"{key} must have finite shape {shape} with magnitude <= 100")
        c[key]=a.tolist()
    return c


def generate(c):
    """Conforming split interface; subdivisions change mesh, not wall dimensions."""
    d=c['dimension'];angle=np.deg2rad(c['angle']);rotation=np.eye(d)
    rotation[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    center=np.array(c['center'][:d]);parts=[];regions=[]
    if c['geometry']=='wall':
        cut=c['width']*(c['fraction']-.5)
        for left,right,label in ((-c['width']/2,cut,'porous'),(cut,c['width']/2,'solid')):
            if left==right:continue
            xs=left+(right-left)*np.linspace(0,1,c['subdivisions']+1)**c['grading']
            for a,b in zip(xs[:-1],xs[1:]):
                lo=np.array([a,-c['thickness']/2]+([-c['depth']/2] if d==3 else []))
                hi=np.array([b,c['thickness']/2]+([c['depth']/2] if d==3 else []))
                if d==2:
                    v=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])
                    cells=[v[[0,1,2]],v[[0,2,3]]]
                else:
                    cells=[]
                    for order in permutations(range(3)):
                        v=[lo.copy()];q=lo.copy()
                        for axis in order:q=q.copy();q[axis]=hi[axis];v.append(q)
                        cells.append(np.array(v))
                for cell in cells:parts.append(cell@rotation.T+center);regions.append(label)
    else:
        # Chord geometry is explicit: refinement changes the represented circle.
        count=c['fibres'] if c['geometry']=='fibres' else 1
        radius=min(c['thickness']/2,c['width']/(3*count)) if count>1 else c['width']/2
        angles=np.arange(c['segments'])*2*np.pi/c['segments']
        for x in np.linspace(-c['width']/2+radius,c['width']/2-radius,count) if count>1 else [0.]:
            ring=np.column_stack([radius*np.cos(angles)+x,radius*np.sin(angles)])
            for a,b in zip(ring,np.roll(ring,-1,axis=0)):
                parts.append(np.array([[x,0.],a,b])@rotation.T+center);regions.append('solid')
    return Geometry(parts,d),np.array(regions),rotation


def requests_for(d,degree):
    # D(s) times affine relative velocity squared needs degree p+2.
    return tuple(dict.fromkeys((*baseline(d),*(Request(a) for a in monomials(d,degree+2)))))


def native_evaluator(c,geometry):
    d=geometry.dimension;req=requests_for(d,c['degree'])
    method='analytic' if c['backend']=='analytic' else 'conditioned_boundary' if d==2 else 'tet_quadrature'
    return Evaluator(geometry,req,kernel=c['kernel'],precision=c['precision'],method=method,
        device=c['device'],preparation=c['preparation'],search=c['search'])


def integrate(c,geometry,queries,*,evaluator=None):
    """Return per-element requested quantities; the browser never invents values."""
    d=geometry.dimension;req=requests_for(d,c['degree'])
    start=time.perf_counter()
    ev=native_evaluator(c,geometry) if evaluator is None else evaluator
    batch=ev.prepare(queries,c['h'],exhaustive=c['exhaustive']);prepared=time.perf_counter()
    batch.launch();batch.numpy();end=time.perf_counter()
    pairs=batch.pairs.numpy()
    return req,batch,pairs,dict(prepare_ms=1000*(prepared-start),action_readback_ms=1000*(end-prepared),
        interactions=batch.count,batch_bytes=batch.bytes,method=ev.method,
        preparation=c['preparation'],search=c['search'],
        search_metrics=getattr(ev.device_search,'last_metrics',None),quadrature=getattr(batch,'quadrature',None))


def contractions(c,queries,req,moments,rotation):
    """Common positive work for supplied affine velocities and polynomial D."""
    d=c['dimension'];n=len(queries);zero=(0,)*d;idx={r:i for i,r in enumerate(req)}
    get=lambda a,ds=():moments[:,idx[Request(tuple(a),tuple(sorted(ds)))]]
    eye=np.eye(d,dtype=int);m0=get(zero);m1=np.stack([get(a) for a in eye],axis=-1)
    m2=np.stack([np.stack([get(a+b) for b in eye],axis=-1) for a in eye],axis=-2)
    grad=np.stack([get(zero,(j,)) for j in range(d)],axis=-1)
    g1=np.stack([np.stack([get(a,(j,)) for j in range(d)],axis=-1) for a in eye],axis=-2)
    H=np.stack([np.stack([get(zero,(i,j)) for j in range(d)],axis=-1) for i in range(d)],axis=-2)
    S=np.array(c['skeleton_gradient'])[:d,:d];L=np.array(c['fluid_gradient'])[:d,:d];R=L-S
    rel=np.array(c['fluid'][:d])-np.array(c['skeleton'][:d])+(queries-np.array(c['center'][:d]))@R.T
    compression=np.einsum('ni,ni->n',rel,grad)-np.einsum('ij,nji->n',S,g1)
    t=rotation[:,0];xi=(queries-np.array(c['center'][:d]))@t/c['width']+.5
    # Exact binomial/multinomial expansion of (1+.25*xi(source))**p.
    coefficients={}
    for alpha in monomials(d,c['degree']):
        k=sum(alpha);coefficient=comb(c['degree'],k)*factorial(k)
        for a in alpha:coefficient/=factorial(a)
        coeff=coefficient*(1+.25*xi)**(c['degree']-k)
        for j,a in enumerate(alpha):coeff=coeff*(.25*t[j]/c['width'])**a
        coefficients[alpha]=coeff
    def weighted(beta):return sum(v*get(tuple(a+b for a,b in zip(alpha,beta))) for alpha,v in coefficients.items())
    wm0=weighted(zero);wm1=np.stack([weighted(a) for a in eye],axis=-1)
    wm2=np.stack([np.stack([weighted(a+b) for b in eye],axis=-1) for a in eye],axis=-2)
    a=np.deg2rad(c['tensor_angle']);axis=np.zeros(d);axis[:2]=[np.cos(a),np.sin(a)]
    tensor=c['resistance']*(np.eye(d)+(c['anisotropy']-1)*np.outer(axis,axis))
    tensor=rotation@tensor@rotation.T
    drag=-(wm0[:,None]*rel+wm1@R.T)@tensor.T
    diss=wm0*np.einsum('ni,ij,nj->n',rel,tensor,rel)+2*np.einsum('ni,ij,nj->n',rel,tensor,wm1@R.T)+np.einsum('ij,jk,kl,nli->n',R.T,tensor,R,wm2)
    integrated=tensor[None,:,:]*wm0[:,None,None]
    mobility=np.linalg.inv(np.eye(d)[None]+c['dt']*integrated)
    driven=np.einsum('nij,nj->ni',mobility,rel)
    normal=rotation[:,1]
    return dict(occupancy=m0,compression=compression,gradient=grad,first=m1,second=m2,
        hessian=H,laplacian=np.trace(H,axis1=-2,axis2=-1),resistance=integrated,
        skeleton_product=(wm0[:,None]*np.array(c['skeleton'][:d])+((queries-np.array(c['center'][:d]))*wm0[:,None]+wm1)@S.T)@tensor.T,
        drag=drag,reaction=-drag,dissipation=diss,mobility=mobility,
        normal_relative=rel@normal,normal_response=driven@normal)


def compute_wall(payload):
    start=time.perf_counter()
    c=validate(payload);d=c['dimension'];geo,labels,rotation=generate(c)
    plot_axes=[0,1] if d==2 else {'xy':[0,1],'xz':[0,2],'yz':[1,2]}[c['slice_plane']]
    vertices=np.concatenate(geo.elements)
    axes=[np.linspace(vertices[:,j].min()-c['h'],vertices[:,j].max()+c['h'],c['resolution']) for j in plot_axes]
    xx,yy=np.meshgrid(*axes);q=np.full((xx.size,d),c['slice'])
    q[:,plot_axes[0]]=xx.ravel();q[:,plot_axes[1]]=yy.ravel()
    if 'queries' in payload:
        q=np.asarray(payload['queries'],float)
        if q.ndim!=2 or q.shape[1]!=d or len(q)>1089 or not np.isfinite(q).all():raise ValueError('finite query array required, at most 1089')
    # Bound peak native sample storage; preserve full row output for diagnostics.
    req=requests_for(d,c['degree']);regional={k:np.zeros((len(q),len(req))) for k in ('solid','porous')}
    selected=[[] for _ in q];timing=[]
    static_start=time.perf_counter();evaluator=native_evaluator(c,geo);static_ms=1000*(time.perf_counter()-static_start)
    for first in range(0,len(q),4 if d==3 else 64):
        req,b,pairs,stats=integrate(c,geo,q[first:first+(4 if d==3 else 64)],evaluator=evaluator)
        timing.append(stats)
        for row,(qi,ei) in enumerate(zip(b.neighbors.query,b.neighbors.element)):
            regional[labels[ei]][first+qi]+=pairs[row]
            selected[first+qi].append(int(ei))
        del b
    if not c['porous_enabled']:
        # Disabling porous coupling retains the solid provider and geometry.
        regional['porous'].fill(0)
    terms={region:contractions(c,q,req,values,rotation) for region,values in regional.items()}
    # Solid volume is an occupancy/continuity provider, never a porous drag law.
    for key in ('resistance','skeleton_product','drag','reaction','dissipation'):
        terms['solid'][key]=np.zeros_like(terms['solid'][key])
    terms['solid']['mobility']=np.broadcast_to(np.eye(d),(len(q),d,d)).copy()
    terms['solid']['normal_response']=terms['solid']['normal_relative'].copy()
    terms['total']={key:terms['solid'][key]+terms['porous'][key] for key in
        ('occupancy','compression','gradient','first','second','hessian','laplacian','resistance','skeleton_product','drag','reaction','dissipation')}
    terms['total']['mobility']=terms['porous']['mobility']
    terms['total']['normal_relative']=terms['porous']['normal_relative']
    terms['total']['normal_response']=terms['porous']['normal_response']
    terms['free']={'occupancy':1-terms['total']['occupancy'],'gradient':-terms['total']['gradient'],
        'first':-terms['total']['first'],'hessian':-terms['total']['hessian'],'laplacian':-terms['total']['laplacian']}
    # Free second moment is supplied explicitly from full-support normalization below.
    from ..edge.kernels import KERNELS
    from ..curvbound.kernels import KERNELS as SHAPES
    radial=float(sum(co*(hi**(p+d+2)-lo**(p+d+2))/(p+d+2)
        for lo,hi,cs in KERNELS[c['kernel']].pieces for p,co in enumerate(cs)))
    scale=(2 if d==2 else 4)*(1 if d==2 else float(SHAPES[c['kernel']].c3_pi/SHAPES[c['kernel']].c2_pi))
    full_second=radial*scale*c['h']**2/d
    terms['free']['second']=full_second*np.eye(d)[None]-terms['total']['second']
    if c['geometry']!='wall':
        # A resolved circular region has no single planar normal. Do not label
        # a fixed Cartesian component as radial permeation through that solid.
        for region in ('solid','porous','total'):
            terms[region].pop('normal_relative',None)
            terms[region].pop('normal_response',None)
    return dict(schema_version=1,kind='fluxbridge-wall',dimension=d,config=c,axes=[a.tolist() for a in axes],plot_axes=plot_axes,
        queries=q.tolist(),shape=[c['resolution']]*2 if 'queries' not in payload else [len(q)],
        elements=[v.tolist() for v in geo.elements],regions=labels.tolist(),selected=selected,
        terms={r:{k:v.tolist() for k,v in fields.items()} for r,fields in terms.items()},
        requests=[dict(alpha=r.alpha,derivative=r.derivative) for r in req],
        moments={r:v.tolist() for r,v in regional.items()},timing=timing,static_prepare_ms=static_ms,elapsed_ms=1000*(time.perf_counter()-start),
        contract=dict(measure='area' if d==2 else 'volume',kernel_dimension=d,support_radius=c['h'],
            precision=c['precision'],device=c['device'],backend=timing[0]['method'] if timing else None,
            preparation=c['preparation'],search=c['search'],geometry_precision='float64',
            occupancy='geometric bulk-region measure, not solid fraction or porosity; the free complement excludes represented porous bulk',
            contraction_precision='host float64; native moment dtype is reported separately',
            units=dict(occupancy='1',first='length',second='length^2',gradient='1/length',
                hessian='1/length^2',laplacian='1/length^2',compression='1/time',resistance='1/time',
                drag='length/time^2 per unit fluid mass',reaction='length/time^2 per unit fluid mass',
                dissipation='length^2/time^3 per unit fluid mass',mobility='1',
                skeleton_product='length/time^2',normal_relative='length/time',normal_response='length/time'),
            compression='integral (u_fluid(query)-u_skeleton(source)) dot grad_query W; prescribed boundary continuity contribution',
            permeation='normal component of a supplied-state local resistance response; no pressure solve',
            resistance='polynomial D in inverse time, not permeability or porosity',
            fibre_flow='transverse 2D cylinder sections; longitudinal response requires an axial formulation or 3D',
            reference='Böttcher et al. (2025), Implicit Incompressible Porous Flow using SPH, arXiv:2504.07739v2'))


def compare_wall(payload):
    """Compare the same probes, geometry dimensions, support and physical fields.

    Mesh refinement of a straight wall preserves geometry; disk/fibre segment
    refinement changes its chord approximation and is labelled separately.
    Precision comparison uses the same numerical method in fp64. Independent
    high-precision fixtures validate the native operators separately.
    """
    base=compute_wall(payload)
    c=base['config'];report=[]
    variants=[('support_search','Exhaustive versus localized support',{'exhaustive':not c['exhaustive']})]
    if c['geometry']=='wall':
        variants.append(('mesh_refinement','Element refinement at fixed wall geometry',
            {'subdivisions':2*c['subdivisions']} if c['subdivisions']<=6 else None))
    else:
        variants.append(('geometry_refinement','Chord refinement changes circle geometry',
            {'segments':2*c['segments']} if c['segments']<=32 else None))
    variants.append(('precision','Same numerical method in fp64',
        {'precision':'float64'} if c['precision']=='float32' else None))
    for key,label,changed in variants:
        if changed is None:
            report.append(dict(key=key,label=label,status='unavailable',reason=(
                'Displayed data already use fp64' if key=='precision' else 'Refinement would exceed the configured geometry limit')))
            continue
        reference=compute_wall({**c,**changed,'queries':base['queries']})
        terms={}
        for region,fields in base['terms'].items():
            terms[region]={}
            for name,value in fields.items():
                a=np.asarray(value,float);b=np.asarray(reference['terms'][region][name],float)
                delta=abs(a-b)
                terms[region][name]=dict(max_absolute=float(delta.max(initial=0)),
                    max_mixed=float((delta/(1+abs(b))).max(initial=0)))
        # Dimensionless moments permit comparing derivative and moment scales.
        scales=np.array([c['h']**Request(**r).scale for r in base['requests']])
        moment_errors={}
        for region in ('solid','porous'):
            a=np.asarray(base['moments'][region])/scales;b=np.asarray(reference['moments'][region])/scales
            moment_errors[region]=float((abs(a-b)/(1+abs(b))).max(initial=0))
        report.append(dict(key=key,label=label,status='measured',changed=changed,
            reference_config=reference['config'],elapsed_ms=reference['elapsed_ms'],
            reference_element_count=len(reference['elements']),terms=terms,normalized_moment_mixed=moment_errors))
    base['comparisons']=dict(rows=report,
        convention='All displayed probes; maximum absolute difference in the displayed physical units and mixed difference |delta|/(1+|reference|). Moment differences first remove h**request.scale.',
        scope='Finite numerical comparisons, not a certified global integration error. Chord refinement changes geometry; wall element refinement does not. No pressure or porosity solve.')
    return base

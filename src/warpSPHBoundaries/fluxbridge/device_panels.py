"""Device-side tetrahedral angular-panel preparation in explicit fp64 geometry.

Numerical directions, weights and cached radii are rounded to the evaluation
dtype at the output boundary. Integer prefix scans allocate the exact rule;
no quadrature arrays are read back to generate nodes or cache indices.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import warp as wp
from .planar_shell import gauss_rule

wp.set_module_options({'enable_backward':False,'fuse_fp':False})
MAX_RULE_BYTES=2**30

TOOLS=r'''
#define FB_AT(...) (*wp::address(__VA_ARGS__))
struct V {double x,y,z;};
auto add=[](V a,V b){return V{a.x+b.x,a.y+b.y,a.z+b.z};};
auto sub=[](V a,V b){return V{a.x-b.x,a.y-b.y,a.z-b.z};};
auto scale=[](V a,double s){return V{a.x*s,a.y*s,a.z*s};};
auto dot=[](V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;};
auto cross=[](V a,V b){return V{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};};
auto length=[&](V a){return sqrt(dot(a,a));};
auto vertex=[&](int f,int j){auto v=FB_AT(facets,start+f,j);return V{(v[0]-query[0])/h,(v[1]-query[1])/h,(v[2]-query[2])/h};};
'''

DISJOINT=r'''
V lower{INFINITY,INFINITY,INFINITY},upper{-INFINITY,-INFINITY,-INFINITY};double span=0;
for(int f=0;f<4;++f)for(int j=0;j<3;++j){
    V v=vertex(f,j);lower=V{wp::min(lower.x,v.x),wp::min(lower.y,v.y),wp::min(lower.z,v.z)};
    upper=V{wp::max(upper.x,v.x),wp::max(upper.y,v.y),wp::max(upper.z,v.z)};
    span=wp::max(span,wp::max(fabs(v.x),wp::max(fabs(v.y),fabs(v.z))));
}
double padding=64*2.2204460492503130808472633361816e-16*(span+1);
V gap{wp::max(0.,wp::max(lower.x,-upper.x)),wp::max(0.,wp::max(lower.y,-upper.y)),wp::max(0.,wp::max(lower.z,-upper.z))};
if(length(gap)>1+padding)return 1;
double nearest=INFINITY,minside=INFINITY,condition=1;
for(int f=0;f<4;++f){
    V a=vertex(f,0),b=vertex(f,1),c=vertex(f,2),ab=sub(b,a),ac=sub(c,a);
    V raw=cross(ab,ac);double magnitude=length(raw);if(!(magnitude>0))return 0;
    V normal=scale(raw,1/magnitude);double distance=dot(normal,a);minside=wp::min(minside,distance);
    double aa=dot(ab,ab),bb=dot(ab,ac),cc=dot(ac,ac),det=aa*cc-bb*bb;
    if(!(det>0))return 0; // uncertain distance classification retains the whole element
    double largest=(aa+cc+sqrt(wp::max(0.,(aa-cc)*(aa-cc)+4*bb*bb)))/2;
    double cond=largest/sqrt(det);condition=wp::max(condition,cond);
    V projected=sub(scale(normal,distance),a);double v=dot(ab,projected),w=dot(ac,projected);
    double s=(cc*v-bb*w)/det,t=(aa*w-bb*v)/det,margin=64*2.2204460492503130808472633361816e-16*cond;
    double best=INFINITY;
    if(s>=-margin&&t>=-margin&&s+t<=1+margin)best=fabs(distance);
    else for(int edge=0;edge<3;++edge){
        V u=vertex(f,edge),v=vertex(f,(edge+1)%3),delta=sub(v,u);
        double along=wp::min(1.,wp::max(0.,-dot(u,delta)/dot(delta,delta)));
        best=wp::min(best,length(add(u,scale(delta,along))));
    }
    nearest=wp::min(nearest,best);
}
padding*=condition;
if(minside>=-padding)return 0;
return nearest>1+padding?1:0;
#undef FB_AT
'''

SECTOR=r'''
using R=FB_REAL;
int face=sector/6,edge=(sector%6)/2,side=(sector%2)?1:-1;
V p=vertex(face,0),p1=vertex(face,1),p2=vertex(face,2);
V normal=cross(sub(p1,p),sub(p2,p));normal=scale(normal,1/length(normal));
double distance=dot(normal,p),c0=fabs(distance);
if(c0==0)return wp::vec_t<2,int64_t>(0,0);
double sign0=distance<0?-1.:1.;
// The regular face potential has no near-plane singularity. Avoid a
// double-logarithmic polar rule when a moderate face lies within roundoff
// of the query: integrate that same potential on the exact triangle.
double face_span=wp::max(length(sub(p1,p)),wp::max(length(sub(p2,p)),length(sub(p2,p1))));
if(c0<=1e-12&&face_span<=4.){
    if(sector%6!=0)return wp::vec_t<2,int64_t>(0,0);
    V ab=sub(p1,p),ac=sub(p2,p);double jacobian=length(cross(ab,ac));
    if(write)for(int i=0;i<order;++i){
        double a=(FB_AT(points,i)+1)/2;V base=add(p,scale(ab,a)),delta=scale(ac,1-a);
        int interval=node_first/order+i;
        FB_AT(info,interval,0)=base.x;FB_AT(info,interval,1)=base.y;FB_AT(info,interval,2)=base.z;
        FB_AT(info,interval,3)=delta.x;FB_AT(info,interval,4)=delta.y;FB_AT(info,interval,5)=delta.z;
        FB_AT(info,interval,9)=jacobian*distance*(1-a)*FB_AT(gauss_weights,i)/4;
        FB_AT(flags,interval,0)=cache_first+i*order;FB_AT(flags,interval,1)=2;FB_AT(flags,interval,2)=1;
    }
    return wp::vec_t<2,int64_t>(order*order,order*order);
}
V av=vertex(face,edge),bv=vertex(face,(edge+1)%3);
V tangent=sub(bv,av);tangent=scale(tangent,1/length(tangent));V outward=cross(tangent,normal);
double z=dot(outward,av),c=fabs(z);
if(c==0)return wp::vec_t<2,int64_t>(0,0);
double sign=z<0?-1.:1.,s0=dot(tangent,av),s1=dot(tangent,bv);
double a=side==1?wp::max(s0,0.):wp::max(-s1,0.),end=side==1?wp::max(s1,0.):wp::max(-s0,0.);
int64_t samples=0,caches=0;int exterior_index=-1;
int phi_iterations=0;
while(a<end){
    if(++phi_iterations>4096)return wp::vec_t<2,int64_t>(-1,-1);
    double b=wp::min(end,wp::max(2*a,c));
    for(int knot=0;knot<(cubic?2:1);++knot){double rad=(cubic&&knot==0)?.5:1.;
        if(rad*rad>c0*c0+c*c){double cut=sqrt(wp::max(0.,rad*rad-c0*c0-c*c));if(a<cut&&cut<b)b=cut;}}
    if(!(b>a))return wp::vec_t<2,int64_t>(-1,-1);
    double left=atan2(a,c),right=atan2(b,c),half=(right-left)/2,last=0;
    for(int i=0;i<order;++i){double phi=(left+right)/2+half*FB_AT(points,i);last=wp::max(last,c/cos(phi));}
    double rho=0,exterior=c0<1?sqrt((1-c0)*(1+c0)):0;int radial_iterations=0;
    while(rho<last){
        if(++radial_iterations>4096)return wp::vec_t<2,int64_t>(-1,-1);
        bool outer=c0>=1||rho>=exterior;
        double high=outer?last:wp::min(last,wp::max(2*rho,c0));
        for(int knot=0;knot<(cubic?2:1);++knot){double rad=(cubic&&knot==0)?.5:1.;
            if(rad>c0){double cut=sqrt((rad-c0)*(rad+c0));if(rho<cut&&cut<high)high=cut;}}
        if(!(high>rho))return wp::vec_t<2,int64_t>(-1,-1);
        int common_index=-1;
        for(int i=0;i<order;++i){
            double phi=(left+right)/2+half*FB_AT(points,i),cs=cos(phi),sn=sin(phi)*side,maxrho=c/cs;
            if(!(maxrho>rho))continue;
            double top=wp::min(high,maxrho),t0=atan2(rho,c0),t1=atan2(top,c0),dh=(t1-t0)/2;
            if(!(dh>0))continue;
            V direction=add(scale(outward,sign*cs),scale(tangent,sn));int cache;
            bool create=true;
            if(outer){if(exterior_index<0){exterior_index=(int)caches;++caches;}else create=false;cache=exterior_index;}
            else if(maxrho>=high){if(common_index<0){common_index=(int)caches;caches+=order;}else create=false;cache=common_index;}
            else{cache=(int)caches;caches+=order;}
            if(samples+order>2000000)return wp::vec_t<2,int64_t>(-1,-1);
            if(write){
                int interval=(node_first+(int)samples)/order;
                FB_AT(info,interval,0)=sign0*normal.x;FB_AT(info,interval,1)=sign0*normal.y;FB_AT(info,interval,2)=sign0*normal.z;
                FB_AT(info,interval,3)=direction.x;FB_AT(info,interval,4)=direction.y;FB_AT(info,interval,5)=direction.z;
                FB_AT(info,interval,6)=(t0+t1)/2;FB_AT(info,interval,7)=dh;FB_AT(info,interval,8)=c0;
                FB_AT(info,interval,9)=sign0*sign*half*FB_AT(gauss_weights,i)*dh;
                FB_AT(flags,interval,0)=cache_first+cache;FB_AT(flags,interval,1)=outer?1:0;FB_AT(flags,interval,2)=create?1:0;
            }
            samples+=order;
        }
        rho=high;
    }
    a=b;
}
return wp::vec_t<2,int64_t>(samples,caches);
#undef FB_AT
'''


@lru_cache(None)
def functions(precision):
    real=wp.float64 if precision=='float64' else wp.float32
    vec=wp.types.vector(length=3,dtype=real)
    @wp.func_native(TOOLS+DISJOINT)
    def disjoint(facets:wp.array2d(dtype=wp.vec3d),start:int,query:wp.vec3d,h:wp.float64)->int: ...
    @wp.func_native((TOOLS+SECTOR).replace('FB_REAL','double' if precision=='float64' else 'float'))
    def sector(facets:wp.array2d(dtype=wp.vec3d),start:int,query:wp.vec3d,h:wp.float64,sector:int,
               order:int,cubic:int,points:wp.array(dtype=wp.float64),gauss_weights:wp.array(dtype=wp.float64),
               write:int,node_first:int,cache_first:int,info:wp.array2d(dtype=wp.float64),flags:wp.array2d(dtype=int))->wp.vec2l: ...
    @wp.kernel
    def reject(facets:wp.array2d(dtype=wp.vec3d),starts:wp.array(dtype=int),elements:wp.array(dtype=int),
               queries:wp.array(dtype=wp.vec3d),supports:wp.array(dtype=wp.float64),skip:wp.array(dtype=int)):
        row=wp.tid();skip[row]=disjoint(facets,starts[elements[row]],queries[row],supports[row])
    @wp.kernel
    def count(facets:wp.array2d(dtype=wp.vec3d),starts:wp.array(dtype=int),elements:wp.array(dtype=int),
              queries:wp.array(dtype=wp.vec3d),supports:wp.array(dtype=wp.float64),skip:wp.array(dtype=int),order:int,cubic:int,
              points:wp.array(dtype=wp.float64),gauss_weights:wp.array(dtype=wp.float64),
              info:wp.array2d(dtype=wp.float64),flags:wp.array2d(dtype=int),
              counts:wp.array(dtype=wp.int64),cache_counts:wp.array(dtype=wp.int64),failure:wp.array(dtype=int)):
        branch=wp.tid();row=branch//24
        if skip[row]==0:
            sizes=sector(facets,starts[elements[row]],queries[row],supports[row],branch%24,order,cubic,
                points,gauss_weights,0,0,0,info,flags)
            if sizes[0]<wp.int64(0):wp.atomic_max(failure,0,1)
            counts[branch+1]=wp.max(sizes[0],wp.int64(0));cache_counts[branch+1]=wp.max(sizes[1],wp.int64(0))
    @wp.kernel
    def fill(facets:wp.array2d(dtype=wp.vec3d),starts:wp.array(dtype=int),elements:wp.array(dtype=int),
             queries:wp.array(dtype=wp.vec3d),supports:wp.array(dtype=wp.float64),skip:wp.array(dtype=int),order:int,cubic:int,
             points:wp.array(dtype=wp.float64),gauss_weights:wp.array(dtype=wp.float64),
             info:wp.array2d(dtype=wp.float64),flags:wp.array2d(dtype=int),
             offsets:wp.array(dtype=wp.int64),cache_offsets:wp.array(dtype=wp.int64),failure:wp.array(dtype=int)):
        branch=wp.tid();row=branch//24
        if skip[row]==0:
            sizes=sector(facets,starts[elements[row]],queries[row],supports[row],branch%24,order,cubic,
                points,gauss_weights,1,int(offsets[branch]),int(cache_offsets[branch]),info,flags)
            if sizes[0]!=offsets[branch+1]-offsets[branch] or sizes[1]!=cache_offsets[branch+1]-cache_offsets[branch]:
                wp.atomic_max(failure,0,2)
    @wp.kernel
    def generate(info:wp.array2d(dtype=wp.float64),flags:wp.array2d(dtype=int),order:int,
                 points:wp.array(dtype=wp.float64),gauss_weights:wp.array(dtype=wp.float64),
                 directions:wp.array(dtype=vec),weights:wp.array(dtype=real),radii:wp.array(dtype=real),indices:wp.array(dtype=int),kinds:wp.array(dtype=int)):
        node=wp.tid();interval=node//order;j=node%order
        if flags[interval,1]==2:
            b=(points[j]+wp.float64(1.))/wp.float64(2.)
            y=wp.vec3d(info[interval,0]+b*info[interval,3],info[interval,1]+b*info[interval,4],info[interval,2]+b*info[interval,5])
            radius=wp.length(y);u=y/radius;index=flags[interval,0]+j
            directions[node]=vec(real(u[0]),real(u[1]),real(u[2]))
            weights[node]=real(info[interval,9]*gauss_weights[j]);radii[index]=real(radius)
            indices[node]=index;kinds[index]=1
            return
        theta=info[interval,6]+info[interval,7]*points[j]
        sn=wp.sin(theta);cs=wp.cos(theta)
        directions[node]=vec(real(info[interval,0]*cs+info[interval,3]*sn),
            real(info[interval,1]*cs+info[interval,4]*sn),real(info[interval,2]*cs+info[interval,5]*sn))
        weights[node]=real(info[interval,9]*gauss_weights[j]*sn)
        outer=flags[interval,1]!=0;index=flags[interval,0]
        if not outer:index=index+j
        indices[node]=index
        if flags[interval,2]!=0 and (not outer or j==0):
            radius=wp.float64(1.)
            if not outer:radius=info[interval,8]/cs
            radii[index]=real(radius)
            kinds[index]=0
    @wp.kernel
    def row_counts(offsets:wp.array(dtype=wp.int64),cache_offsets:wp.array(dtype=wp.int64),
                   nodes:wp.array(dtype=int),profiles:wp.array(dtype=int),tiles:wp.array(dtype=int),failure:wp.array(dtype=int)):
        row=wp.tid();nodes[row]=int(offsets[24*row]);profiles[row]=int(cache_offsets[24*row])
        if row+1<nodes.shape[0]:
            size=offsets[24*(row+1)]-offsets[24*row]
            if size>wp.int64(2000000):wp.atomic_max(failure,0,3)
            tiles[row+1]=(int(size)+255)//256
    @wp.kernel
    def tile_rows(tile_offsets:wp.array(dtype=int),rows:wp.array(dtype=int)):
        row=wp.tid()
        for tile in range(tile_offsets[row],tile_offsets[row+1]):rows[tile]=row
    return SimpleNamespace(real=real,vec=vec,reject=reject,count=count,fill=fill,generate=generate,row_counts=row_counts,tile_rows=tile_rows)


def samples(batch,local,queries,supports,order):
    """Count, scan, allocate and fill one rule on the selected Warp device."""
    from warp.utils import array_scan
    ev=batch.evaluator;api=functions(ev.precision);device=ev.device;kw=dict(device=device)
    count=batch.count;branches=24*count
    if not count or not ev.host_plan.requests:
        return dict(directions=wp.empty(0,dtype=api.vec,**kw),weights=wp.empty(0,dtype=api.real,**kw),
            radii=wp.empty(0,dtype=api.real,**kw),profile_index=wp.empty(0,dtype=int,**kw),
            profile_kind=wp.empty(0,dtype=int,**kw),
            node_offsets=wp.zeros(count+1,dtype=int,**kw),profile_offsets=wp.zeros(count+1,dtype=int,**kw),
            tile_offsets=wp.zeros(count+1,dtype=int,**kw),tile_rows=wp.empty(0,dtype=int,**kw),tiles=0)
    faces=wp.array(local,dtype=wp.vec3d,**kw);q=wp.array(queries,dtype=wp.vec3d,**kw);h=wp.array(supports,dtype=wp.float64,**kw)
    starts=wp.array(ev.geometry.starts,dtype=int,**kw);skip=wp.empty(count,dtype=int,**kw)
    points,weights=gauss_rule(order).T
    points=wp.array(np.ascontiguousarray(points),dtype=wp.float64,**kw)
    gauss=wp.array(np.ascontiguousarray(weights),dtype=wp.float64,**kw)
    counts=wp.zeros(branches+1,dtype=wp.int64,**kw);cached=wp.zeros_like(counts)
    offsets=wp.empty_like(counts);cache_offsets=wp.empty_like(counts);failure=wp.zeros(1,dtype=int,**kw)
    empty_info=wp.empty((0,10),dtype=wp.float64,**kw);empty_flags=wp.empty((0,3),dtype=int,**kw)
    wp.launch(api.reject,count,inputs=[faces,starts,batch.pe,q,h,skip],device=device)
    shared=[faces,starts,batch.pe,q,h,skip,order,int(ev.host_plan.kernel=='cubic'),points,gauss]
    wp.launch(api.count,branches,inputs=shared+[empty_info,empty_flags,counts,cached,failure],device=device)
    array_scan(counts,offsets);array_scan(cached,cache_offsets)
    # Two device-to-host scalar totals determine allocation. No node/radius data
    # is copied back for panel construction, sorting or index generation.
    totals=wp.empty(2,dtype=wp.int64,device='cpu',pinned=device.is_cuda)
    wp.copy(totals,offsets,dest_offset=0,src_offset=branches,count=1)
    wp.copy(totals,cache_offsets,dest_offset=1,src_offset=branches,count=1)
    wp.synchronize_device(device)
    total,unique=map(int,totals.numpy())
    if failure.numpy()[0]:raise ValueError('device tetrahedral panel/sample budget exceeded')
    if max(total,unique)>2**31-1:raise ValueError('device quadrature indices exceed int32 capacity')
    nodes=wp.empty(count+1,dtype=int,**kw);profile_offsets=wp.empty_like(nodes);tile_counts=wp.zeros_like(nodes);tile_offsets=wp.empty_like(nodes)
    wp.launch(api.row_counts,count+1,inputs=[offsets,cache_offsets,nodes,profile_offsets,tile_counts,failure],device=device)
    array_scan(tile_counts,tile_offsets)
    tile_total=wp.empty(1,dtype=int,device='cpu',pinned=device.is_cuda)
    wp.copy(tile_total,tile_offsets,src_offset=count,count=1);wp.synchronize_device(device);tiles=int(tile_total.numpy()[0])
    if failure.numpy()[0]:raise ValueError('device tetrahedral sample budget exceeded per interaction')
    scalars=unique*len(ev.host_plan.profiles)+tiles*len(ev.host_plan.requests)
    width=8 if ev.precision=='float64' else 4
    storage=width*(4*total+unique+4*count)+4*(total+unique+tiles+3*(count+1))+max(width*scalars,92*(total//order))
    if storage>MAX_RULE_BYTES:raise ValueError('device tet quadrature storage exceeds the 1 GiB per-rule budget')
    result=dict(directions=wp.empty(total,dtype=api.vec,**kw),weights=wp.empty(total,dtype=api.real,**kw),
        radii=wp.empty(unique,dtype=api.real,**kw),profile_index=wp.empty(total,dtype=int,**kw),
        profile_kind=wp.empty(unique,dtype=int,**kw),
        node_offsets=nodes,profile_offsets=profile_offsets,tile_offsets=tile_offsets,
        tile_rows=wp.empty(tiles,dtype=int,**kw),tiles=tiles)
    info=wp.empty((total//order,10),dtype=wp.float64,**kw);flags=wp.empty((total//order,3),dtype=int,**kw)
    wp.launch(api.fill,branches,inputs=shared+[info,flags,offsets,cache_offsets,failure],device=device)
    if total:
        wp.launch(api.generate,total,inputs=[info,flags,order,points,gauss,result['directions'],result['weights'],result['radii'],result['profile_index'],result['profile_kind']],device=device)
    wp.launch(api.tile_rows,count,inputs=[tile_offsets,result['tile_rows']],device=device)
    if failure.numpy()[0]:raise ValueError('device quadrature count/fill disagreement')
    return result

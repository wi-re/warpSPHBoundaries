"""Reusable whole-element BVH search, with explicit fp64 geometric filtering.

Warp's BVH stores outward-rounded fp32 boxes. Count, traversal, scans, image
enumeration and deterministic sorting run on the selected CPU/CUDA device.
The current batch interface consumes a frozen host Adjacency, so CSR arrays
are read back; this is preparation, not a GPU-only moving-particle pipeline.
"""
import numpy as np
import warp as wp
from .geometry import Adjacency, fingerprint, queries_and_supports

wp.set_module_options({'enable_backward': False, 'fuse_fp': False})
MAX_IMAGES = 4096
MAX_SEARCH_BYTES = 2**30


@wp.kernel
def image_ranges(q: wp.array(dtype=wp.vec3d), h: wp.array(dtype=wp.float64),
                 periods: wp.vec3d, low: wp.vec3d, high: wp.vec3d,
                 ranges: wp.array2d(dtype=wp.int64), error: wp.array(dtype=int)):
    i = wp.tid()
    size = wp.int64(1)
    for axis in range(3):
        first = wp.float64(0.0)
        last = wp.float64(0.0)
        if periods[axis] > wp.float64(0.0):
            a = (q[i][axis]-h[i]-high[axis])/periods[axis]
            b = (q[i][axis]+h[i]-low[axis])/periods[axis]
            # Include boundary images conservatively under division rounding.
            margin = wp.float64(1.4210854715202004e-14)*wp.max(wp.float64(1.0),wp.max(wp.abs(a),wp.abs(b)))
            first = wp.ceil(a-margin)
            last = wp.floor(b+margin)
        if not wp.isfinite(first) or not wp.isfinite(last) or wp.max(wp.abs(first),wp.abs(last)) > wp.float64(4503599627370495.0):
            wp.atomic_max(error,0,1)
            first = wp.float64(1.0)
            last = wp.float64(0.0)
        count = wp.max(wp.int64(0),wp.int64(last-first+wp.float64(1.0)))
        if count > wp.int64(MAX_IMAGES):
            wp.atomic_max(error,0,2)
            count = wp.int64(0)
        size = size*count
        if size > wp.int64(MAX_IMAGES):
            wp.atomic_max(error,0,2)
            size = wp.int64(0)
        ranges[i,2*axis] = wp.int64(first)
        ranges[i,2*axis+1] = wp.int64(count)


@wp.func_native(r'''
#define AT(a,i) (*reinterpret_cast<decltype(a.data)>((char*)a.data+(i)*a.strides[0]))
#define RANGE(a,i,j) (*reinterpret_cast<int64_t*>((char*)a.data+(i)*a.strides[0]+(j)*a.strides[1]))
auto outward=[](double x,bool up){
    float v=static_cast<float>(wp::min(1e38,wp::max(-1e38,x)));
    union {float f; uint32_t u;} a; a.f=v;
    if(v==0){a.u=up?1u:0x80000001u;}
    else if((v>0)==up){++a.u;}else{--a.u;}
    return a.f;
};
int64_t found=0,serial=0;
for(int64_t x=0;x<RANGE(ranges,qi,1);++x)
for(int64_t y=0;y<RANGE(ranges,qi,3);++y)
for(int64_t z=0;z<RANGE(ranges,qi,5);++z,++serial){
    wp::vec3d shift(double(RANGE(ranges,qi,0)+x)*periods[0],
        double(RANGE(ranges,qi,2)+y)*periods[1],double(RANGE(ranges,qi,4)+z)*periods[2]);
    wp::vec3d world=AT(q,qi)-shift,local=world-origin;
    double radius=AT(h,qi);
    wp::vec3f lo,hi;
    for(int a=0;a<3;++a){lo[a]=outward(local[a]-radius,false);hi[a]=outward(local[a]+radius,true);}
    auto query=wp::bvh_query_aabb(tree,lo,hi,-1);int eid=0;
    while(wp::bvh_query_next(query,eid,FLT_MAX)){
        double squared=0;
        for(int a=0;a<3;++a){double delta=wp::max(0.,wp::max(AT(lower,eid)[a]-world[a],world[a]-AT(upper,eid)[a]))/radius;
            squared+=delta*delta;}
        // AABB filtering is conservative; never remove individual faces.
        if(squared>1.+1.4210854715202004e-14)continue;
        if(write){int64_t pos=start+found;
            AT(keys,pos)=(int64_t(qi)*elements+eid)*4096+serial;
            AT(values,pos)=static_cast<int>(pos);AT(shifts,pos)=shift;}
        ++found;
    }
}
return found;
#undef AT
#undef RANGE
''', adj_snippet='')
def traverse(tree: wp.uint64, qi: int, elements: int,
             q: wp.array(dtype=wp.vec3d), h: wp.array(dtype=wp.float64),
             lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
             origin: wp.vec3d, periods: wp.vec3d, ranges: wp.array2d(dtype=wp.int64),
             start: wp.int64, write: bool, keys: wp.array(dtype=wp.int64),
             values: wp.array(dtype=int), shifts: wp.array(dtype=wp.vec3d)) -> wp.int64: ...


@wp.kernel
def count_pairs(tree: wp.uint64, elements: int, q: wp.array(dtype=wp.vec3d),
                h: wp.array(dtype=wp.float64), lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
                origin: wp.vec3d, periods: wp.vec3d, ranges: wp.array2d(dtype=wp.int64),
                counts: wp.array(dtype=wp.int64), keys: wp.array(dtype=wp.int64),
                values: wp.array(dtype=int), shifts: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    counts[i] = traverse(tree,i,elements,q,h,lower,upper,origin,periods,ranges,wp.int64(0),False,keys,values,shifts)


@wp.kernel
def fill_pairs(tree: wp.uint64, elements: int, q: wp.array(dtype=wp.vec3d),
               h: wp.array(dtype=wp.float64), lower: wp.array(dtype=wp.vec3d), upper: wp.array(dtype=wp.vec3d),
               origin: wp.vec3d, periods: wp.vec3d, ranges: wp.array2d(dtype=wp.int64),
               offsets: wp.array(dtype=wp.int64), keys: wp.array(dtype=wp.int64),
               values: wp.array(dtype=int), shifts: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    traverse(tree,i,elements,q,h,lower,upper,origin,periods,ranges,offsets[i],True,keys,values,shifts)


@wp.kernel
def gather_pairs(elements: int, keys: wp.array(dtype=wp.int64), values: wp.array(dtype=int),
                 source: wp.array(dtype=wp.vec3d), queries: wp.array(dtype=int),
                 owners: wp.array(dtype=int), shifts: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    pair = keys[i]//wp.int64(MAX_IMAGES)
    queries[i] = int(pair//wp.int64(elements))
    owners[i] = int(pair%wp.int64(elements))
    shifts[i] = source[values[i]]


class DeviceSearch:
    """Static whole-element boxes/BVH, reused for new query/support snapshots."""
    def __init__(self, geometry, *, device='cpu'):
        self.geometry = geometry
        self.revision = geometry.revision
        self.device = wp.get_device(device)
        self.dimension = geometry.dimension
        self.count = len(geometry.elements)
        if self.count>2**31-1 or self.count*72>MAX_SEARCH_BYTES:
            raise ValueError('device search static arrays exceed 1 GiB budget or int32 element range')
        pad = lambda x: np.pad(x,((0,0),(0,3-self.dimension)))
        bounds = geometry.bounds
        self.origin = pad(bounds[:,0])[0] if self.count else np.zeros(3)
        lower = pad(bounds[:,0]); upper = pad(bounds[:,1])
        self.low = wp.vec3d(lower.min(axis=0)) if self.count else wp.vec3d(0.)
        self.high = wp.vec3d(upper.max(axis=0)) if self.count else wp.vec3d(0.)
        self.lower = wp.array(lower,dtype=wp.vec3d,device=self.device)
        self.upper = wp.array(upper,dtype=wp.vec3d,device=self.device)
        relative = np.stack((lower-self.origin,upper-self.origin))
        if not np.isfinite(relative).all() or np.any(abs(relative)>1e37):
            raise ValueError('BVH local geometry exceeds finite fp32 box range')
        lo = np.nextafter(relative[0].astype(np.float32),np.float32(-np.inf))
        hi = np.nextafter(relative[1].astype(np.float32),np.float32(np.inf))
        self.lowers = wp.array(lo,dtype=wp.vec3f,device=self.device)
        self.uppers = wp.array(hi,dtype=wp.vec3f,device=self.device)
        self.bvh = wp.Bvh(self.lowers,self.uppers,constructor='lbvh' if self.device.is_cuda else 'sah') if self.count else None
        self.last_metrics = None

    @property
    def array_bytes(self):
        """Owned array capacity; Warp BVH's internal allocation is additional."""
        return sum(a.capacity for a in (self.lower,self.upper,self.lowers,self.uppers))

    def prepare(self, queries, supports, *, periods=None):
        if self.geometry.revision != self.revision:
            raise ValueError('stale device search geometry; rebuild the BVH')
        q,h = queries_and_supports(self.geometry,queries,supports)
        d = self.dimension; n = len(q)
        periodic = tuple(periods) if periods is not None else (0.,)*d
        if len(periodic)!=d or any(not np.isfinite(v) or v<0 for v in periodic):
            raise ValueError('periods must be finite nonnegative lengths; zero is nonperiodic')
        if not n or not self.count:
            arrays = [np.empty(0,np.int32),np.empty(0,np.int32),np.empty((0,d)),np.zeros(n+1,np.int32)]
            self.last_metrics = dict(interactions=0,query_array_bytes=0,readback_bytes=arrays[-1].nbytes)
        else:
            if n*self.count*MAX_IMAGES > 2**63-1 or n>2**31-1:
                raise ValueError('device search sort-key/index range exceeded')
            base_bytes = n*(24+8+48+8+8)+20
            if base_bytes>MAX_SEARCH_BYTES:
                raise ValueError('device search storage exceeds 1 GiB budget')
            padded = np.pad(q,((0,0),(0,3-d)))
            qd = wp.array(padded,dtype=wp.vec3d,device=self.device)
            hd = wp.array(h,dtype=wp.float64,device=self.device)
            ranges = wp.empty((n,6),dtype=wp.int64,device=self.device)
            error = wp.zeros(1,dtype=int,device=self.device)
            periods3 = wp.vec3d(*(periodic+(0.,)*(3-d)))
            wp.launch(image_ranges,n,inputs=[qd,hd,periods3,self.low,self.high,ranges,error],device=self.device)
            code = int(error.numpy()[0])
            if code:
                raise ValueError('periodic image integer range exceeded' if code==1 else 'device search exceeds 4096 periodic images per query')
            counts = wp.zeros(n+1,dtype=wp.int64,device=self.device)
            offsets = wp.empty(n+1,dtype=wp.int64,device=self.device)
            keys = wp.empty(0,dtype=wp.int64,device=self.device)
            values = wp.empty(0,dtype=int,device=self.device)
            temporary = wp.empty(0,dtype=wp.vec3d,device=self.device)
            inputs = [self.bvh.id,self.count,qd,hd,self.lower,self.upper,wp.vec3d(self.origin),periods3,ranges]
            wp.launch(count_pairs,n,inputs=inputs+[counts,keys,values,temporary],device=self.device)
            wp.utils.array_scan(counts,offsets,inclusive=False)
            host_offsets = offsets.numpy()
            size = int(host_offsets[-1])
            if size>2**31-1 or base_bytes+size*80>MAX_SEARCH_BYTES:
                raise ValueError('device search storage exceeds 1 GiB budget or int32 pair range')
            keys = wp.empty(2*size,dtype=wp.int64,device=self.device)
            values = wp.empty(2*size,dtype=int,device=self.device)
            temporary = wp.empty(size,dtype=wp.vec3d,device=self.device)
            pq = wp.empty(size,dtype=int,device=self.device); pe = wp.empty_like(pq)
            shifts = wp.empty(size,dtype=wp.vec3d,device=self.device)
            if size:
                wp.launch(fill_pairs,n,inputs=inputs+[offsets,keys,values,temporary],device=self.device)
                wp.utils.radix_sort_pairs(keys,values,count=size)
                wp.launch(gather_pairs,size,inputs=[self.count,keys,values,temporary,pq,pe,shifts],device=self.device)
            arrays = [pq.numpy(),pe.numpy(),shifts.numpy()[:,:d].copy(),host_offsets.astype(np.int32)]
            self.last_metrics = dict(interactions=size,query_array_bytes=base_bytes+size*80,
                readback_bytes=32*size+8*(n+1)+4,host_csr_bytes=sum(a.nbytes for a in arrays),
                geometry_precision='float64',bvh_precision='outward-rounded float32',
                max_images=MAX_IMAGES,host_csr_readback=True)
        for a in arrays: a.flags.writeable=False
        return Adjacency(*arrays,fingerprint(q,h),self.revision,periodic)

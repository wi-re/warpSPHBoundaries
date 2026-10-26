"""Allocation-free analytic device evaluators, explicit fp32/fp64 factories.

The caller owns disjoint scratch rows. Static plans contain no quadrature.
Kernel derivatives are explicit outputs; reverse-mode shape AD is not exposed.
"""
from functools import lru_cache
from types import SimpleNamespace
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@lru_cache(None)
def functions(precision="float64",*,method="analytic",nodes=16):
    if method=="tet_quadrature":
        from .tet_quadrature import functions as tet_functions
        return tet_functions(precision,nodes)
    if method=="conditioned_boundary":
        from .conditioned2d import functions as conditioned_functions
        return conditioned_functions(precision,nodes)
    if method!="analytic":raise ValueError("unknown integration method")
    if precision=="float32_compensated":
        from .paired import functions as paired_functions
        return paired_functions()
    if precision not in ("float32","float64"):
        raise ValueError("precision must be float32, float64 or float32_compensated")
    real=wp.float64 if precision=="float64" else wp.float32
    vec=wp.types.vector(length=3,dtype=real)
    poly=wp.types.vector(length=12,dtype=real)
    cuts=wp.types.vector(length=4,dtype=real)

    @wp.struct
    class Plan:
        f: wp.array2d(dtype=int)
        dep: wp.array2d(dtype=int)
        fr: wp.array(dtype=real)
        b: wp.array2d(dtype=int)
        br: wp.array(dtype=real)
        v: wp.array2d(dtype=int)
        vr: wp.array(dtype=real)
        term_kind: wp.array(dtype=int)
        term_index: wp.array(dtype=int)
        term_value: wp.array(dtype=real)
        offsets: wp.array(dtype=int)
        dimension: int
        surface: int
        compact: int

    @wp.func
    def power(x: real,n: int):
        result=real(1.0)
        count=n
        if count<0:
            count=-count
        for k in range(count):
            result=result*x
        if n<0:
            result=real(1.0)/result
        return result

    @wp.func
    def asinh(x: real):
        value=wp.log(wp.abs(x)+wp.sqrt(x*x+real(1.0)))
        if x<real(0.0):
            value=-value
        return value

    @wp.func
    def primitive(m: int,s: real,c: real):
        c2=c*c
        r=wp.sqrt(s*s+c2)
        result=real(0.0)
        if m==-3:
            result=s/(c2*r)
        elif m==-2:
            result=wp.atan2(s,c)/c
        elif m==-1:
            result=asinh(s/c)
        elif m%2==0:
            result=s
            for k in range(2,m+1,2):
                result=(s*power(r,k)+real(k)*c2*result)/real(k+1)
        else:
            tail=real(0.0)
            if c>real(0.0):
                tail=c2*asinh(s/c)
            result=(s*r+tail)/real(2.0)
            for k in range(3,m+1,2):
                result=(s*power(r,k)+real(k)*c2*result)/real(k+1)
        return result

    @wp.func
    def binomial(n: int,k: int):
        out=real(1.0)
        for i in range(k):
            out=out*real(n-i)/real(i+1)
        return out

    @wp.func
    def monomial_primitive(j: int,n: int,s: real,c: real):
        result=real(0.0)
        half=j//2
        r=wp.sqrt(s*s+c*c)
        for k in range(half+1):
            coef=binomial(half,k)*power(-c*c,half-k)
            if coef!=real(0.0):
                value=real(0.0)
                m=n+2*k
                if j%2==0:
                    value=primitive(m,s,c)
                else:
                    if m==-2:
                        value=wp.log(r)
                    else:
                        value=power(r,m+2)/real(m+2)
                result=result+coef*value
        return result

    @wp.func
    def line(a: vec,b: vec,alpha: wp.vec3i,n: int,R: real):
        tangent=wp.normalize(b-a)
        lo=wp.dot(tangent,a)
        hi=wp.dot(tangent,b)
        offset=a-lo*tangent
        c=wp.length(offset)
        value=real(0.0)
        if R<real(0.0) or c<R:
            if R>real(0.0):
                L=wp.sqrt(wp.max(real(0.0),(R-c)*(R+c)))
                lo=wp.max(lo,-L)
                hi=wp.min(hi,L)
            if hi>lo and (n>=0 or c>real(0.0) or alpha[0]+alpha[1]+alpha[2]>0):
                coefficients=poly()
                coefficients[0]=real(1.0)
                degree=int(0)
                for axis in range(3):
                    for k in range(alpha[axis]):
                        # Descending update preserves the previous polynomial.
                        for jj in range(degree+2):
                            j=degree+1-jj
                            v=real(0.0)
                            if j>0:
                                v=coefficients[j-1]*tangent[axis]
                            if j<=degree:
                                v=v+coefficients[j]*offset[axis]
                            coefficients[j]=v
                        degree=degree+1
                for j in range(degree+1):
                    if coefficients[j]!=real(0.0):
                        value=value+coefficients[j]*(monomial_primitive(j,n,hi,c)-monomial_primitive(j,n,lo,c))
        return value

    @wp.func
    def K(m: int,s: real,w: real,a: real):
        b=wp.abs(w)
        c=wp.sqrt(w*w+a*a)
        result=real(0.0)
        if m%2==0:
            result=wp.atan2(s,b)/b
            for k in range(2,m+1,2):
                result=primitive(k-2,s,c)+a*a*result
        else:
            r=wp.sqrt(s*s+c*c)
            if a>real(0.0):
                result=wp.atan2(a*s,b*r)/(a*b)
            else:
                result=s/(b*b*r)
            for k in range(1,m+1,2):
                result=primitive(k-2,s,c)+a*a*result
        return result

    @wp.func
    def scalar_face(p: vec,q: vec,r: vec,normal: vec,n: int,R: real):
        a=wp.abs(wp.dot(normal,p))
        total=real(0.0)
        if (R<real(0.0) or a<R) and (n!=-3 or a>real(0.0)):
            for edge in range(3):
                v=p
                u=q
                if edge==1:
                    v=q
                    u=r
                if edge==2:
                    v=r
                    u=p
                tangent=wp.normalize(u-v)
                mu=wp.cross(tangent,normal)
                w=wp.dot(mu,v)
                lo=wp.dot(tangent,v)
                hi=wp.dot(tangent,u)
                if w!=real(0.0):
                    points=cuts()
                    points[0]=lo
                    count=int(1)
                    c2=a*a+w*w
                    if R>real(0.0) and R*R>c2:
                        L=wp.sqrt(R*R-c2)
                        if lo < -L and -L < hi:
                            points[count]=-L
                            count=count+1
                        if lo < L and L < hi:
                            points[count]=L
                            count=count+1
                    points[count]=hi
                    for k in range(count):
                        l=points[k]
                        u2=points[k+1]
                        mid=(l+u2)/real(2.0)
                        # atan2 avoids subtracting near-equal endpoint angles.
                        d0=wp.atan2(wp.abs(w)*(u2-l),w*w+l*u2)/wp.abs(w)
                        term=real(0.0)
                        if R<real(0.0) or mid*mid+c2<R*R:
                            term=K(n+2,u2,w,a)-K(n+2,l,w,a)-power(a,n+2)*d0
                        else:
                            term=(power(R,n+2)-power(a,n+2))*d0
                        total=total+w*term/real(n+2)
        return total

    @wp.func
    def element(query: vec,h: real,facets: wp.array2d(dtype=vec),first: int,last: int,
                plan: Plan,work: wp.array2d(dtype=real),row: int):
        nf=plan.f.shape[0]
        nb=plan.b.shape[0]
        nv=plan.v.shape[0]
        for k in range(nb):
            work[row,nf+k]=real(0.0)
        if plan.surface!=0:
            # A clipped-out open triangle still owes initialized face outputs.
            for k in range(nf):
                work[row,k]=real(0.0)
        if plan.compact!=0 and plan.surface==0:
            angle=real(0.0)
            for face in range(first,last):
                a=(facets[face,0]-query)/h
                b=(facets[face,1]-query)/h
                c=(facets[face,2]-query)/h
                omega=real(0.0)
                if plan.dimension==2:
                    det=a[0]*b[1]-a[1]*b[0]
                    if det!=real(0.0):
                        omega=wp.atan2(det,wp.dot(a,b))
                else:
                    det=wp.dot(a,wp.cross(b-a,c-a))
                    if det!=real(0.0):
                        den=wp.length(a)*wp.length(b)*wp.length(c)+wp.dot(a,b)*wp.length(c)+wp.dot(b,c)*wp.length(a)+wp.dot(c,a)*wp.length(b)
                        omega=real(2.0)*wp.atan2(det,den)
                angle=angle+omega
            for k in range(nb):
                if plan.b[k,0]<0:
                    n=plan.b[k,1]
                    work[row,nf+k]=angle*power(plan.br[k],n+plan.dimension)/real(n+plan.dimension)
        for face in range(first,last):
            p=(facets[face,0]-query)/h
            q=(facets[face,1]-query)/h
            r=(facets[face,2]-query)/h
            if plan.compact!=0:
                delta=vec()
                for j in range(3):
                    low=wp.min(p[j],wp.min(q[j],r[j]))
                    high=wp.max(p[j],wp.max(q[j],r[j]))
                    delta[j]=wp.max(real(0.0),wp.max(low,-high))
                if wp.dot(delta,delta)>=real(1.0):
                    continue
            normal=vec()
            if plan.dimension==3:
                normal=wp.normalize(wp.cross(q-p,r-p))
            else:
                tangent=wp.normalize(q-p)
                normal=vec(tangent[1],-tangent[0],real(0.0))
            z=wp.dot(normal,p)
            for k in range(nf):
                n=plan.f[k,0]
                axis=plan.f[k,1]
                R=plan.fr[k]
                alpha=wp.vec3i(plan.f[k,3],plan.f[k,4],plan.f[k,5])
                val=real(0.0)
                if plan.dimension==2:
                    val=line(p,q,alpha,n,R)
                elif axis<0:
                    val=scalar_face(p,q,r,normal,n,R)
                else:
                    beta=wp.vec3i(plan.f[k,6],plan.f[k,7],plan.f[k,8])
                    val=z*normal[axis]*work[row,plan.f[k,2]]
                    edge_term=real(0.0)
                    for e in range(3):
                        v=p
                        u=q
                        if e==1:
                            v=q
                            u=r
                        if e==2:
                            v=r
                            u=p
                        mu=wp.cross(wp.normalize(u-v),normal)
                        edge_term=edge_term+mu[axis]*(line(v,u,beta,n+2,R)-power(R,n+2)*line(v,u,beta,0,R))
                    for j in range(3):
                        coeff=plan.dep[k,3*j]
                        if coeff!=0:
                            proj=-normal[axis]*normal[j]
                            if axis==j:
                                proj=proj+real(1.0)
                            edge_term=edge_term-real(coeff)*proj*(work[row,plan.dep[k,3*j+1]]-power(R,n+2)*work[row,plan.dep[k,3*j+2]])
                    val=val+edge_term/real(n+2)
                work[row,k]=val
            if plan.surface!=0:
                # Surface batches have one triangle; no orientation factor.
                pass
            else:
                for k in range(nb):
                    axis=plan.b[k,0]
                    val=real(0.0)
                    if axis<0:
                        if z!=real(0.0):
                            n=plan.b[k,1]
                            tail=-work[row,plan.b[k,4]]
                            if plan.compact==0:
                                tail=tail+work[row,plan.b[k,3]]
                            val=z*(work[row,plan.b[k,2]]+power(plan.br[k],n+plan.dimension)*tail)/real(n+plan.dimension)
                    else:
                        val=normal[axis]*work[row,plan.b[k,2]]
                    work[row,nf+k]=work[row,nf+k]+val
        for k in range(nv):
            axis=plan.v[k,0]
            val=work[row,nf+plan.v[k,1]]
            if axis>=0:
                n=plan.v[k,6]
                R=plan.vr[k]
                val=val-power(R,n+2)*work[row,nf+plan.v[k,2]]
                coeff=plan.v[k,5]
                if coeff!=0:
                    val=val-real(coeff)*(work[row,nf+nb+plan.v[k,3]]-power(R,n+2)*work[row,nf+nb+plan.v[k,4]])
                val=val/real(n+2)
            work[row,nf+nb+k]=val

    @wp.func
    def output(plan: Plan,work: wp.array2d(dtype=real),row: int,column: int):
        val=real(0.0)
        compensation=real(0.0)
        nf=plan.f.shape[0]
        nb=plan.b.shape[0]
        for k in range(plan.offsets[column],plan.offsets[column+1]):
            kind=plan.term_kind[k]
            idx=plan.term_index[k]
            if kind==1:
                idx=idx+nf+nb
            elif kind==2:
                idx=idx+nf
            # Neumaier summation: profile coefficients can alternate strongly.
            term=plan.term_value[k]*work[row,idx]
            new=val+term
            if wp.abs(val)>=wp.abs(term):
                compensation=compensation+((val-new)+term)
            else:
                compensation=compensation+((term-new)+val)
            val=new
        return val+compensation

    @wp.kernel
    def batch(queries: wp.array(dtype=vec),supports: wp.array(dtype=real),
              facets: wp.array2d(dtype=vec),starts: wp.array(dtype=int),
              pair_q: wp.array(dtype=int),pair_e: wp.array(dtype=int),shifts: wp.array(dtype=vec),
              plan: Plan,scales: wp.array(dtype=int),work: wp.array2d(dtype=real),out: wp.array2d(dtype=real)):
        row=wp.tid()
        qi=pair_q[row]
        ei=pair_e[row]
        h=supports[qi]
        element(queries[qi]-shifts[row],h,facets,starts[ei],starts[ei+1],plan,work,row)
        for col in range(out.shape[1]):
            out[row,col]=output(plan,work,row,col)*power(h,scales[col])

    return SimpleNamespace(real=real,vec=vec,Plan=Plan,work_dtype=real,coefficient_dtype=real,field_dtype=real,output_cache=False,local_geometry=False,element=element,output=output,batch=batch,power=power)

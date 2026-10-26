"""Local C++/CUDA arithmetic shared by native paired-fp32 device functions.

Each snippet receives the same local definitions. No float64 arithmetic,
quadrature, or external headers are used. FMA is explicitly single precision.
"""
ARITHMETIC = r'''
struct DS {
    float h, l;
    CUDA_CALLABLE DS(float value=0.0f):h(value),l(0.0f) {}
    CUDA_CALLABLE DS(float hi,float lo):h(hi),l(lo) {}
    CUDA_CALLABLE DS(wp::vec2f value):h(value[0]),l(value[1]) {}
    CUDA_CALLABLE static DS norm(float a,float b) {
        float s=a+b; return DS(s,b-(s-a));
    }
    CUDA_CALLABLE DS operator-() const { return DS(-h,-l); }
    CUDA_CALLABLE DS operator+(DS b) const {
        float s=h+b.h, v=s-h;
        float e=((h-(s-v))+(b.h-v))+(l+b.l);
        return norm(s,e);
    }
    CUDA_CALLABLE DS operator-(DS b) const { return *this+(-b); }
    CUDA_CALLABLE DS operator*(DS b) const {
        float p=h*b.h;
        float e=fmaf(h,b.h,-p)+(h*b.l+l*b.h);
        return norm(p,e);
    }
    CUDA_CALLABLE DS operator/(DS b) const {
        float q=h/b.h;
        DS rem=*this-b*DS(q);
        return DS(q)+DS(rem.h/b.h);
    }
    CUDA_CALLABLE bool operator<(DS b) const {return h<b.h || (h==b.h && l<b.l);}
    CUDA_CALLABLE bool operator>(DS b) const {return b<*this;}
    CUDA_CALLABLE bool operator<=(DS b) const {return !(b<*this);}
    CUDA_CALLABLE bool operator>=(DS b) const {return !(*this<b);}
    CUDA_CALLABLE bool operator==(DS b) const {return h==b.h && l==b.l;}
    CUDA_CALLABLE bool operator!=(DS b) const {return !(*this==b);}
    CUDA_CALLABLE wp::vec2f pair() const {return wp::vec2f(h,l);}
    CUDA_CALLABLE float value() const {return h+l;}
    CUDA_CALLABLE static DS abs(DS a) {return a<DS(0)?-a:a;}
    CUDA_CALLABLE static DS min(DS a,DS b) {return a<b?a:b;}
    CUDA_CALLABLE static DS max(DS a,DS b) {return a>b?a:b;}
    CUDA_CALLABLE static DS power(DS a,int n) {
        DS r(1); int count=n<0?-n:n;
        for(int i=0;i<count;++i) r=r*a;
        return n<0?DS(1)/r:r;
    }
    CUDA_CALLABLE static DS sqrt(DS a) {
        if(a==DS(0)) return DS(0);
        DS r(sqrtf(a.h)); return r+(a-r*r)/(DS(2)*r);
    }
    CUDA_CALLABLE static DS log(DS a) {
        int e=static_cast<int>(floorf(log2f(a.h)));
        DS m=a*DS(ldexpf(1.0f,-e));
        DS z=(m-DS(1))/(m+DS(1)), z2=z*z, term=z, sum=z;
        for(int k=1;k<20;++k) {term=term*z2; sum=sum+term/DS(2*k+1);}
        return DS(2)*sum+DS(e)*DS(0.693147182464599609375f,-1.904654299957768e-9f);
    }
    CUDA_CALLABLE static DS atan2(DS y,DS x) {
        if(x==DS(0) && y==DS(0)) return DS(0);
        float angle=atan2f(y.h,x.h);
        int quadrant=static_cast<int>(floorf(angle/1.5707963267948966f+0.5f));
        DS r=DS(angle)-DS(quadrant)*DS(1.57079637050628662109375f,-4.371139000186243e-8f);
        DS r2=r*r, st=r, ct(1), sn=r, cs(1);
        for(int k=1;k<=9;++k) {
            st=-st*r2/DS((2*k)*(2*k+1)); sn=sn+st;
            ct=-ct*r2/DS((2*k-1)*(2*k)); cs=cs+ct;
        }
        int q=(quadrant%4+4)%4;
        DS s=sn,c=cs;
        if(q==1){s=cs;c=-sn;}
        if(q==2){s=-sn;c=-cs;}
        if(q==3){s=-cs;c=sn;}
        DS t=(y*c-x*s)/(x*c+y*s);
        return DS(angle)+t-t*t*t/DS(3);
    }
    CUDA_CALLABLE static DS asinh(DS x) {
        DS a=abs(x); DS v;
        if(a>DS(1.0e10f)) v=log(a)+DS(0.693147182464599609375f,-1.904654299957768e-9f);
        else v=log(a+sqrt(a*a+DS(1)));
        return x<DS(0)?-v:v;
    }
};
struct V {
    DS v[3];
    CUDA_CALLABLE V() {}
    CUDA_CALLABLE V(wp::vec3f a) {for(int j=0;j<3;++j)v[j]=DS(a[j]);}
    CUDA_CALLABLE V(DS a,DS b,DS c) {v[0]=a;v[1]=b;v[2]=c;}
    CUDA_CALLABLE DS& operator[](int j){return v[j];}
    CUDA_CALLABLE DS operator[](int j)const{return v[j];}
    CUDA_CALLABLE V operator+(V b) const {return V(v[0]+b[0],v[1]+b[1],v[2]+b[2]);}
    CUDA_CALLABLE V operator-(V b) const {return V(v[0]-b[0],v[1]-b[1],v[2]-b[2]);}
    CUDA_CALLABLE V operator*(DS b) const {return V(v[0]*b,v[1]*b,v[2]*b);}
    CUDA_CALLABLE V operator/(DS b) const {return V(v[0]/b,v[1]/b,v[2]/b);}
    CUDA_CALLABLE static DS dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
    CUDA_CALLABLE static V cross(V a,V b){return V(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]);}
    CUDA_CALLABLE DS length() const{return DS::sqrt(dot(*this,*this));}
    CUDA_CALLABLE V normalized() const{return *this/length();}
};
'''

PRIMITIVES = r'''
auto primitive=[&](int m,DS s,DS c) {
    DS c2=c*c, r=DS::sqrt(s*s+c2), result;
    if(m==-3) return s/(c2*r);
    if(m==-2) return DS::atan2(s,c)/c;
    if(m==-1) return DS::asinh(s/c);
    if(m%2==0) {
        result=s;
        for(int k=2;k<=m;k+=2) result=(s*DS::power(r,k)+DS(k)*c2*result)/DS(k+1);
    } else {
        DS tail;
        if(c>DS(0)) tail=c2*DS::asinh(s/c);
        result=(s*r+tail)/DS(2);
        for(int k=3;k<=m;k+=2) result=(s*DS::power(r,k)+DS(k)*c2*result)/DS(k+1);
    }
    return result;
};
auto binomial=[&](int n,int k) {
    DS out(1); for(int i=0;i<k;++i) out=out*DS(n-i)/DS(i+1); return out;
};
auto monomial=[&](int j,int n,DS s,DS c) {
    DS sum, r=DS::sqrt(s*s+c*c); int half=j/2;
    for(int k=0;k<=half;++k) {
        DS coef=binomial(half,k)*DS::power(-c*c,half-k);
        if(coef!=DS(0)) {
            int m=n+2*k; DS value;
            if(j%2==0) value=primitive(m,s,c);
            else value=m==-2?DS::log(r):DS::power(r,m+2)/DS(m+2);
            sum=sum+coef*value;
        }
    }
    return sum;
};
auto line=[&](V a,V b,int ax,int ay,int az,int n,DS R) {
    V tangent=(b-a).normalized();
    DS lo=V::dot(tangent,a),hi=V::dot(tangent,b);
    V offset=a-tangent*lo; DS c=offset.length(),sum;
    if(R<DS(0) || c<R) {
        if(R>DS(0)) {
            DS L=DS::sqrt(DS::max(DS(0),(R-c)*(R+c)));
            lo=DS::max(lo,-L);hi=DS::min(hi,L);
        }
        if(hi>lo && (n>=0 || c>DS(0) || ax+ay+az>0)) {
            DS coef[12];coef[0]=DS(1);int degree=0;int alpha[3]={ax,ay,az};
            for(int axis=0;axis<3;++axis) for(int k=0;k<alpha[axis];++k) {
                for(int j=degree+1;j>=0;--j) {
                    DS v;
                    if(j>0) v=coef[j-1]*tangent[axis];
                    if(j<=degree) v=v+coef[j]*offset[axis];
                    coef[j]=v;
                }
                ++degree;
            }
            for(int j=0;j<=degree;++j) if(coef[j]!=DS(0))
                sum=sum+coef[j]*(monomial(j,n,hi,c)-monomial(j,n,lo,c));
        }
    }
    return sum;
};
auto K=[&](int m,DS s,DS w,DS a) {
    DS b=DS::abs(w),c=DS::sqrt(w*w+a*a),result;
    if(m%2==0) {
        result=DS::atan2(s,b)/b;
        for(int k=2;k<=m;k+=2) result=primitive(k-2,s,c)+a*a*result;
    } else {
        DS r=DS::sqrt(s*s+c*c);
        result=a>DS(0)?DS::atan2(a*s,b*r)/(a*b):s/(b*b*r);
        for(int k=1;k<=m;k+=2) result=primitive(k-2,s,c)+a*a*result;
    }
    return result;
};
auto scalar_face=[&](V p,V q,V r,V normal,int n,DS R) {
    DS a=DS::abs(V::dot(normal,p)),total;
    if((R<DS(0) || a<R) && (n!=-3 || a>DS(0))) {
        V vertices[3]={p,q,r};
        for(int e=0;e<3;++e) {
            V v=vertices[e],u=vertices[(e+1)%3],tangent=(u-v).normalized();
            V mu=V::cross(tangent,normal);
            DS w=V::dot(mu,v),lo=V::dot(tangent,v),hi=V::dot(tangent,u);
            if(w!=DS(0)) {
                DS points[4];points[0]=lo;int count=1;DS c2=a*a+w*w;
                if(R>DS(0) && R*R>c2) {
                    DS L=DS::sqrt(R*R-c2);
                    if(lo< -L && -L<hi)points[count++]=-L;
                    if(lo<L && L<hi)points[count++]=L;
                }
                points[count]=hi;
                for(int k=0;k<count;++k) {
                    DS l=points[k],u2=points[k+1],mid=(l+u2)/DS(2);
                    DS d0=DS::atan2(DS::abs(w)*(u2-l),w*w+l*u2)/DS::abs(w),term;
                    if(R<DS(0) || mid*mid+c2<R*R)
                        term=K(n+2,u2,w,a)-K(n+2,l,w,a)-DS::power(a,n+2)*d0;
                    else term=(DS::power(R,n+2)-DS::power(a,n+2))*d0;
                    total=total+w*term/DS(n+2);
                }
            }
        }
    }
    return total;
};
'''

ELEMENT = r'''
#define FB_LOAD(...) (*wp::address(__VA_ARGS__))
auto F=[&](int k,int j){return FB_LOAD(plan.f,k,j);};
auto B=[&](int k,int j){return FB_LOAD(plan.b,k,j);};
auto U=[&](int k,int j){return FB_LOAD(plan.v,k,j);};
auto W=[&](int k){return DS(FB_LOAD(work,row,k));};
auto store=[&](int k,DS v){wp::array_store(work,row,k,v.pair());};
int nf=plan.f.shape[0],nb=plan.b.shape[0],nv=plan.v.shape[0];
for(int k=0;k<nb;++k)store(nf+k,DS(0));
if(plan.surface)for(int k=0;k<nf;++k)store(k,DS(0));
if(plan.compact && !plan.surface) {
    DS angle;
    for(int face=first;face<last;++face) {
        V a=(V(FB_LOAD(facets,face,0))-V(query))/DS(h);
        V b=(V(FB_LOAD(facets,face,1))-V(query))/DS(h);
        V c=(V(FB_LOAD(facets,face,2))-V(query))/DS(h);
        DS omega;
        if(plan.dimension==2) {
            DS det=a[0]*b[1]-a[1]*b[0];
            if(det!=DS(0))omega=DS::atan2(det,V::dot(a,b));
        } else {
            DS det=V::dot(a,V::cross(b-a,c-a));
            if(det!=DS(0)) {
                DS den=a.length()*b.length()*c.length()+V::dot(a,b)*c.length()+V::dot(b,c)*a.length()+V::dot(c,a)*b.length();
                omega=DS(2)*DS::atan2(det,den);
            }
        }
        angle=angle+omega;
    }
    for(int k=0;k<nb;++k)if(B(k,0)<0) {
        int n=B(k,1);
        store(nf+k,angle*DS::power(DS(FB_LOAD(plan.br,k)),n+plan.dimension)/DS(n+plan.dimension));
    }
}
for(int face=first;face<last;++face) {
    V p=(V(FB_LOAD(facets,face,0))-V(query))/DS(h);
    V q=(V(FB_LOAD(facets,face,1))-V(query))/DS(h);
    V r=(V(FB_LOAD(facets,face,2))-V(query))/DS(h);
    if(plan.compact) {
        V delta;
        for(int j=0;j<3;++j) {
            DS low=DS::min(p[j],DS::min(q[j],r[j]));
            DS high=DS::max(p[j],DS::max(q[j],r[j]));
            delta[j]=DS::max(DS(0),DS::max(low,-high));
        }
        if(V::dot(delta,delta)>=DS(1))continue;
    }
    V normal;
    if(plan.dimension==3)normal=V::cross(q-p,r-p).normalized();
    else {V t=(q-p).normalized();normal=V(t[1],-t[0],DS(0));}
    DS z=V::dot(normal,p);
    for(int k=0;k<nf;++k) {
        int n=F(k,0),axis=F(k,1);DS R(FB_LOAD(plan.fr,k)),value;
        if(plan.dimension==2)value=line(p,q,F(k,3),F(k,4),F(k,5),n,R);
        else if(axis<0)value=scalar_face(p,q,r,normal,n,R);
        else {
            value=z*normal[axis]*W(F(k,2));
            DS edge_term;V vertices[3]={p,q,r};
            for(int e=0;e<3;++e) {
                V v=vertices[e],u=vertices[(e+1)%3];
                V mu=V::cross((u-v).normalized(),normal);
                edge_term=edge_term+mu[axis]*(line(v,u,F(k,6),F(k,7),F(k,8),n+2,R)-DS::power(R,n+2)*line(v,u,F(k,6),F(k,7),F(k,8),0,R));
            }
            for(int j=0;j<3;++j) {
                int coeff=FB_LOAD(plan.dep,k,3*j);
                if(coeff) {
                    DS proj=-normal[axis]*normal[j];
                    if(axis==j)proj=proj+DS(1);
                    edge_term=edge_term-DS(coeff)*proj*(W(FB_LOAD(plan.dep,k,3*j+1))-DS::power(R,n+2)*W(FB_LOAD(plan.dep,k,3*j+2)));
                }
            }
            value=value+edge_term/DS(n+2);
        }
        store(k,value);
    }
    if(!plan.surface)for(int k=0;k<nb;++k) {
        int axis=B(k,0);DS value;
        if(axis<0) {
            if(z!=DS(0)) {
                int n=B(k,1);DS tail=-W(B(k,4));
                if(!plan.compact)tail=tail+W(B(k,3));
                value=z*(W(B(k,2))+DS::power(DS(FB_LOAD(plan.br,k)),n+plan.dimension)*tail)/DS(n+plan.dimension);
            }
        } else value=normal[axis]*W(B(k,2));
        store(nf+k,W(nf+k)+value);
    }
}
for(int k=0;k<nv;++k) {
    int axis=U(k,0);DS value=W(nf+U(k,1));
    if(axis>=0) {
        int n=U(k,6),coeff=U(k,5);DS R(FB_LOAD(plan.vr,k));
        value=value-DS::power(R,n+2)*W(nf+U(k,2));
        if(coeff)value=value-DS(coeff)*(W(nf+nb+U(k,3))-DS::power(R,n+2)*W(nf+nb+U(k,4)));
        value=value/DS(n+2);
    }
    store(nf+nb+k,value);
}
#undef FB_LOAD
'''

PROFILE = r'''
auto evaluate=[&](int column) {
    DS result;
    int nf=plan.f.shape[0],nb=plan.b.shape[0];
    for(int k=*wp::address(plan.offsets,column);k<*wp::address(plan.offsets,column+1);++k) {
        int kind=*wp::address(plan.term_kind,k),index=*wp::address(plan.term_index,k);
        if(kind==1)index+=nf+nb;
        if(kind==2)index+=nf;
        result=result+DS(*wp::address(plan.term_value,k))*DS(*wp::address(work,row,index));
    }
    return result;
};
'''
CACHE = r'''
int start=plan.f.shape[0]+plan.b.shape[0]+plan.v.shape[0];
for(int column=0;column<plan.offsets.shape[0]-1;++column)
    wp::array_store(work,row,start+column,evaluate(column).pair());
'''
CACHED_OUTPUT = r'''
int start=plan.f.shape[0]+plan.b.shape[0]+plan.v.shape[0];
return *wp::address(work,row,start+column);
'''

CONTRACT = r'''
V delta=V(query);
int start=plan.f.shape[0]+plan.b.shape[0]+plan.v.shape[0];
for(int c=0;c<pairs.shape[1];++c) {
    DS total;
    for(int t=0;t<terms.shape[0];++t) {
        int a=*wp::address(terms,t,0),column=*wp::address(terms,t,1);
        DS coefficient(*wp::address(coefficients,ei,a,c));
        if(coefficient!=DS(0)) {
            DS weight=DS(*wp::address(factors,t))*DS(*wp::address(work,row,start+column))*DS::power(DS(h),*wp::address(scales,column));
            for(int j=0;j<3;++j)weight=weight*DS::power(delta[j],*wp::address(terms,t,2+j));
            total=total+coefficient*weight;
        }
    }
    wp::array_store(pairs,row,c,total.value());
}
'''
TRANSPOSE = r'''
int start=plan.f.shape[0]+plan.b.shape[0]+plan.v.shape[0];
DS total;
for(int k=*wp::address(offsets,ei);k<*wp::address(offsets,ei+1);++k) {
    int row=*wp::address(reverse,k),qi=*wp::address(pq,row);
    V delta=V(*wp::address(queries,row));
    DS weight;
    for(int t=0;t<terms.shape[0];++t)if(*wp::address(terms,t,0)==a) {
        int col=*wp::address(terms,t,1);
        DS value=DS(*wp::address(factors,t))*DS(*wp::address(work,row,start+col))*DS::power(DS(*wp::address(supports,qi)),*wp::address(scales,col));
        for(int j=0;j<3;++j)value=value*DS::power(delta[j],*wp::address(terms,t,2+j));
        weight=weight+value;
    }
    total=total+DS(*wp::address(volumes,qi))*DS(*wp::address(loads,qi,c))*weight;
}
wp::array_store(out,ei,a,c,total.value());
'''

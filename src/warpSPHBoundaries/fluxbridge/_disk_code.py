"""Circle boundary quadrature with an analytic exterior-flux correction.

Unlike the analytic polygon/radial families this method deliberately integrates
curved boundary profiles numerically. Quadrature order is explicit in the plan.
"""
from ._radial_code import arithmetic


def definitions(precision):
    code=arithmetic(precision)
    if precision=="float32_compensated":
        return code+r'''
S pi(3.1415927410125732421875f,-8.742278000372485e-8f);
auto root=[&](S x){return S::sqrt(x);};
auto angle_of=[&](S y,S x){return S::atan2(y,x);};
auto trig=[&](S angle,S& sn,S& cs) {
    int k=static_cast<int>(floorf(angle.h/1.5707963267948966f+0.5f));
    S r=angle-S(k)*S(1.57079637050628662109375f,-4.371139000186243e-8f);
    S r2=r*r,st=r,ct(1),s=r,c(1);
    for(int j=1;j<=9;++j) {
        st=-st*r2/S((2*j)*(2*j+1));s=s+st;
        ct=-ct*r2/S((2*j-1)*(2*j));c=c+ct;
    }
    int q=(k%4+4)%4;
    sn=s;cs=c;
    if(q==1){sn=c;cs=-s;}
    if(q==2){sn=-s;cs=-c;}
    if(q==3){sn=-c;cs=s;}
};
'''
    if precision=="float32":
        return code+r'''
S pi(3.14159265358979323846f);
auto root=[&](S x){return S(::sqrtf(x.value()));};
auto angle_of=[&](S y,S x){return S(::atan2f(y.value(),x.value()));};
// Only 0 <= angle <= pi/2 occurs here. Avoid libdevice's unreachable
// large-argument sine/cosine reduction, which includes fp64 instructions.
auto trig=[&](S angle,S& sn,S& cs) {
    int k=angle.value()>0.7853981633974483f?1:0;
    S r=angle-S(k)*S(1.5707963267948966f),r2=r*r,st=r,ct(1),s=r,c(1);
    for(int j=1;j<=6;++j) {
        st=-st*r2/S((2*j)*(2*j+1));s=s+st;
        ct=-ct*r2/S((2*j-1)*(2*j));c=c+ct;
    }
    sn=s;cs=c;if(k==1){sn=c;cs=-s;}
};
'''
    suffix=""
    return code+r'''
S pi(PI);
auto root=[&](S x){return S(::SQRT(x.value()));};
auto angle_of=[&](S y,S x){return S(::ATAN2(y.value(),x.value()));};
auto trig=[&](S x,S& sn,S& cs){sn=S(::SIN(x.value()));cs=S(::COS(x.value()));};
'''.replace('PI','3.14159265358979323846'+suffix).replace('SQRT','sqrt'+suffix).replace('ATAN2','atan2'+suffix).replace('SIN','sin'+suffix).replace('COS','cos'+suffix)


ELEMENT=r'''
S H(h),d=S(distance)/H,R=S(radius)/H,D=R+d;
auto load=[&](auto a,int i) {auto v=wp::load(wp::address(a,i));return S(v[0],v[1]);};
auto profile=[&](S r,S* result) {
    for(int j=0;j<5;++j)result[j]=S(0);
    if(r<S(1)) {
        for(int p=0;p<plan.bounds.shape[0];++p) {
            auto bounds=wp::load(wp::address(plan.bounds,p));
            if(!(r<S(bounds[0])) && r<S(bounds[1])) {
                S x=S(2)*r/S(bounds[1])-S(1);
                for(int j=0;j<5;++j) {
                    int count=wp::load(wp::address(plan.lengths,p,j));S b1,b2;
                    for(int k=count-1;k>=1;--k) {
                        auto c=wp::load(wp::address(plan.coefficients,p,j,k));
                        S next=S(2)*x*b1-b2+S(c[0],c[1]);b2=b1;b1=next;
                    }
                    auto c=wp::load(wp::address(plan.coefficients,p,j,0));
                    result[j]=x*b1-b2+S(c[0],c[1]);
                }
                for(int j=0;j<2;++j) {
                    auto c=wp::load(wp::address(plan.offsets,p,j));S off(c[0],c[1]);
                    if(off!=S(0))result[j+2]=result[j+2]+off/(r*r);
                }
                auto c=wp::load(wp::address(plan.offsets,p,2));S linear(c[0],c[1]);
                if(linear!=S(0))result[4]=result[4]+linear/r;
            }
        }
    } else {
        result[2]=load(plan.constants,0)/(r*r);
        result[3]=load(plan.constants,1)/(r*r);
    }
};
S sums[9],residuals[9];
auto add=[&](int j,S value) {
    S term=value-residuals[j],next=sums[j]+term;
    residuals[j]=(next-sums[j])-term;sums[j]=next;
};
if(d<=-S(1)) {
    sums[0]=S(1);sums[2]=load(plan.constants,2);sums[3]=sums[2];
    sums[5]=S(1);sums[6]=S(1);
} else if(d<S(1)) {
    if(D==S(0)) {
        S values[5];profile(R,values);S area=pi*R*R;
        sums[0]=S(2)*area*values[2];
        sums[2]=area*(values[1]-S(2)*values[3]);sums[3]=sums[2];
        sums[5]=sums[0]-area*values[0];sums[6]=sums[5];
        sums[7]=area*values[4];sums[8]=sums[7];
    } else {
        auto arc=[&](S knot) {
            if(knot<=S::abs(d))return S(0);
            if(!(knot<S(2)*R+d))return pi;
            S u=(knot-S::abs(d))*(knot+S::abs(d));
            S v=(S(2)*R+d-knot)*(S(2)*R+d+knot);
            return S(2)*angle_of(root(u),root(v));
        };
        S end=arc(S(1)),points[32];points[0]=S(0);points[1]=end;int count=2;
        for(int p=0;p<plan.bounds.shape[0]-1;++p) {
            auto bound=wp::load(wp::address(plan.bounds,p));S value=arc(S(bound[1]));
            if(value>S(0) && value<end)points[count++]=value;
        }
        if(d!=S(0)) {
            S scale=S::max(S::abs(d)/root(D*R),end*S(1.0e-8f));
            for(int k=0;k<24 && scale<end;++k) {points[count++]=scale;scale=scale*S(4);}
        }
        for(int i=1;i<count;++i) {
            S value=points[i];int j=i-1;
            while(j>=0 && value<points[j]){points[j+1]=points[j];--j;}
            points[j+1]=value;
        }
        // The regular radial potentials have exterior 1/r^2 tails. Preserve
        // their entire far-circle flux analytically; never discard that arc.
        if(S(2)*R+d>S(1)) {
            S sn,cs;trig(end/S(2),sn,cs);
            S yx=-d-S(2)*R*sn*sn,yy=S(2)*R*sn*cs;
            S flux=S(2)*angle_of(yy,-yx);
            add(0,flux*load(plan.constants,0));
            add(2,-flux*load(plan.constants,1));
            add(3,-flux*load(plan.constants,1));
        }
        for(int p=0;p<count-1;++p) {
            S lo=points[p],hi=points[p+1],half=(hi-lo)/S(2),mid=(lo+hi)/S(2);
            if(hi>lo)for(int k=0;k<plan.nodes.shape[0];++k) {
                S phi=mid+half*load(plan.nodes,k),sn,cs;trig(phi/S(2),sn,cs);
                S nx=S(1)-S(2)*sn*sn,ny=S(2)*sn*cs;
                S yx=-d-S(2)*R*sn*sn,yy=R*ny;
                S r=root(d*d+S(4)*D*R*sn*sn),ndoty=-d+S(2)*D*sn*sn;
                S values[5];profile(r,values);
                S weight=S(2)*R*half*load(plan.weights,k);
                add(0,weight*ndoty*values[2]);
                add(1,weight*nx*values[1]);
                add(2,weight*(ny*yy*values[1]-ndoty*values[3]));
                add(3,weight*(nx*yx*values[1]-ndoty*values[3]));
                add(4,-weight*nx*values[0]);
                add(5,-weight*ny*yy*values[0]);
                add(6,-weight*nx*yx*values[0]);
                add(7,weight*ny*yy*values[4]);
                add(8,weight*nx*yx*values[4]);
            }
        }
        sums[5]=sums[5]+sums[0];sums[6]=sums[6]+sums[0];
    }
}
int scales[9]={0,1,2,2,-1,0,0,-2,-2};
for(int j=0;j<9;++j)
    wp::store(wp::address(work,row,j),(sums[j]*S::power(H,scales[j])).value());
'''

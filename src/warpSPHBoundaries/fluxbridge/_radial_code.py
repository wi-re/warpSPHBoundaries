"""Analytic radial primitives for the complete 3D axisymmetric baseline.

The angular cap is mu >= t along the inward axis. Factoring 1-t in an interval
coordinate avoids the nearly concentric sphere's cancelling powers of R/D.
Polynomial parts use Bernstein products. Cubic's outer Laurent derivatives
also use a bounded analytic beta series. No quadrature enters this code.
"""


def arithmetic(precision):
    if precision=="float32_compensated":
        from ._paired_code import ARITHMETIC
        return ARITHMETIC+"\nusing S=DS;\n"
    base="double" if precision=="float64" else "float"
    return r'''
struct S {
    BASE x;
    CUDA_CALLABLE S(BASE v=0):x(v) {}
    CUDA_CALLABLE S(BASE hi,BASE lo):x(hi+lo) {}
    CUDA_CALLABLE S operator-()const{return S(-x);}
    CUDA_CALLABLE S operator+(S b)const{return S(x+b.x);}
    CUDA_CALLABLE S operator-(S b)const{return S(x-b.x);}
    CUDA_CALLABLE S operator*(S b)const{return S(x*b.x);}
    CUDA_CALLABLE S operator/(S b)const{return S(x/b.x);}
    CUDA_CALLABLE bool operator<(S b)const{return x<b.x;}
    CUDA_CALLABLE bool operator>(S b)const{return x>b.x;}
    CUDA_CALLABLE bool operator<=(S b)const{return x<=b.x;}
    CUDA_CALLABLE bool operator==(S b)const{return x==b.x;}
    CUDA_CALLABLE bool operator!=(S b)const{return x!=b.x;}
    CUDA_CALLABLE BASE value()const{return x;}
    CUDA_CALLABLE static S min(S a,S b){return a<b?a:b;}
    CUDA_CALLABLE static S max(S a,S b){return a>b?a:b;}
    CUDA_CALLABLE static S abs(S a){return a<S(0)?-a:a;}
    CUDA_CALLABLE static S power(S a,int n){
        S result(1);int count=n<0?-n:n;
        for(int k=0;k<count;++k)result=result*a;
        return n<0?S(1)/result:result;
    }
};
'''.replace("BASE",base)


ELEMENT = r'''
S H(h),d=S(distance)/H,R=S(radius)/H,D=R+d;
S full_end=S::min(S(1),S::max(S(0),-d));
S cap_lo=S::abs(d),cap_hi(1);
if(kind==1)cap_hi=S::min(S(1),S(2)*R+d);
// Integrate products of linear factors in Bernstein form on [lo,hi].
// Positive endpoint factors give positive coefficients, avoiding cancellation
// both at the sphere centre and at kernel/support tangencies.
auto integrate=[&](int radial,int shifted,int angular,S end,S lo,S hi) {
    S width=hi-lo,coeff[20];coeff[0]=S(1);int degree=0;
    for(int group=0;group<4;++group) {
        int repeats=0;S f0,f1;
        if(group==0){repeats=radial;f0=lo;f1=hi;}
        if(group==1){repeats=shifted;f0=end-lo;f1=end-hi;}
        if(group==2 && angular>0) {
            repeats=angular;
            if(kind==0){f0=lo-d;f1=hi-d;}
            else {f0=((R-lo)+D)/(S(2)*D);f1=((R-hi)+D)/(S(2)*D);}
        }
        if(group==3 && kind==1 && angular>0){repeats=angular;f0=lo-d;f1=hi-d;}
        for(int k=0;k<repeats;++k) {
            for(int j=degree+1;j>=0;--j) {
                S value;
                if(j<=degree)value=coeff[j]*f0*S(degree+1-j)/S(degree+1);
                if(j>0)value=value+coeff[j-1]*f1*S(j)/S(degree+1);
                coeff[j]=value;
            }
            ++degree;
        }
    }
    S sum;
    if(radial>=0) {
        for(int j=0;j<=degree;++j)sum=sum+coeff[j];
    } else {
        // Only cubic's outer Q profile reaches this branch: lo>=1/2,
        // s=-radial<=2, eta=(hi-lo)/hi<=1/2. Expand the rational primitive
        // around hi, integrate each Bernstein beta moment exactly. The
        // 64-term positive series has scalar tail < 8e-18 before hi^-s.
        int s=-radial;S eta=width/hi;
        for(int j=0;j<=degree;++j) {
            S term(1),series(1);
            for(int k=0;k<63;++k) {
                term=term*eta*S(s+k)/S(k+1)*S(degree-j+k+1)/S(degree+k+2);
                S next=series+term;
                // eta<=1/2 and s<=2 make these positive terms nonincreasing.
                // Once a raw-fp32 addition rounds away, every later one does
                // too. This preserves the fixed-64-loop floating result.
                if(FB_BETA_EARLY_EXIT && next==series)break;
                series=next;
            }
            sum=sum+coeff[j]*series*S::power(hi,-s);
        }
    }
    return width*sum/S(degree+1);
};
auto moment=[&](int m,int denominator,int tangent,int radial,int shifted,S piece_lo,S piece_hi) {
    int p=m+2-denominator+radial;
    S sum,lo=piece_lo,hi=S::min(piece_hi,full_end);
    if(hi>lo && m!=1)
        sum=(m==0?S(4):S(4)/S(3))*integrate(p,shifted,0,piece_hi,lo,hi);
    lo=S::max(piece_lo,cap_lo);hi=S::min(piece_hi,cap_hi);
    if(hi>lo) {
        S j1=integrate(p-1,shifted,1,piece_hi,lo,hi);
        if(m==0)sum=sum+S(2)*j1;
        if(m==1)sum=sum+S(2)*j1-integrate(p-2,shifted,2,piece_hi,lo,hi);
        if(m==2 && tangent==0)sum=sum+S(2)*j1-S(2)*integrate(p-2,shifted,2,piece_hi,lo,hi)+S(2)/S(3)*integrate(p-3,shifted,3,piece_hi,lo,hi);
        if(m==2 && tangent==1)sum=sum+integrate(p-2,shifted,2,piece_hi,lo,hi)-integrate(p-3,shifted,3,piece_hi,lo,hi)/S(3);
    }
    return sum;
};
S sums[9],residuals[9];
for(int k=0;k<plan.powers.shape[0];++k) {
    int n=wp::load(wp::address(plan.powers,k));
    int radial=wp::load(wp::address(plan.radial_powers,k));
    int profile=wp::load(wp::address(plan.profiles,k));
    auto bounds=wp::load(wp::address(plan.bounds,k));
    auto coefficient=wp::load(wp::address(plan.coefficients,k));
    S c(coefficient[0],coefficient[1]),lo(bounds[0]),hi(bounds[1]),terms[9];
    if(profile==0) {
        terms[0]=c*moment(0,0,0,radial,n,lo,hi);
        terms[1]=-c*moment(1,0,0,radial,n,lo,hi);
        terms[2]=c*moment(2,0,1,radial,n,lo,hi);
        terms[3]=c*moment(2,0,0,radial,n,lo,hi);
    }
    if(profile==1 || profile==3) {
        int denominator=profile==3?1:0;
        terms[4]=c*moment(1,denominator,0,radial,n,lo,hi);
        terms[5]=-c*moment(2,denominator,1,radial,n,lo,hi);
        terms[6]=-c*moment(2,denominator,0,radial,n,lo,hi);
        terms[7]=c*moment(0,denominator,0,radial,n,lo,hi);
        terms[8]=terms[7];
    }
    if(profile==2 || profile==4) {
        int denominator=profile==4?3:1;
        terms[7]=c*moment(2,denominator,1,radial,n,lo,hi);
        terms[8]=c*moment(2,denominator,0,radial,n,lo,hi);
    }
    for(int j=0;j<9;++j) {
        S term=terms[j]-residuals[j],next=sums[j]+term;
        residuals[j]=(next-sums[j])-term;sums[j]=next;
    }
}
int scales[9]={0,1,2,2,-1,0,0,-2,-2};
for(int j=0;j<9;++j)
    wp::store(wp::address(work,row,j),(sums[j]*S::power(H,scales[j])).value());
'''

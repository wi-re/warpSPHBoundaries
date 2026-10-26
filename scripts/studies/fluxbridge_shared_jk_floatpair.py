"""Experimental two-float arithmetic; elementary-function seeds stay fp32.

This diagnoses recurrence/accumulation precision without any device fp64.
It is not a production arithmetic library or a universal double substitute.
"""
import math
import re
import numpy as np

HEADER=r'''
struct Pair {
 using Base=float;
 float hi,lo;
 Pair():hi(0),lo(0){}
 Pair(float x):hi(x),lo(0){}
 Pair(float h,float l):hi(h),lo(l){}
 operator Base()const{return hi+lo;}
 Pair operator-()const{return Pair(-hi,-lo);}
 Pair operator+(Pair b)const{
  float s=hi+b.hi,v=s-hi,e=(hi-(s-v))+(b.hi-v);
  e+=lo+b.lo;float h=s+e;return Pair(h,e-(h-s));
 }
 Pair operator-(Pair b)const{return *this+(-b);}
 Pair operator*(Pair b)const{
  float p=hi*b.hi,e=::fmaf(hi,b.hi,-p);
  e+=hi*b.lo+lo*b.hi; e+=lo*b.lo;
  float h=p+e;return Pair(h,e-(h-p));
 }
 Pair operator/(Pair b)const{
  float q=hi/b.hi;Pair r=*this-b*Pair(q);
  return Pair(q)+Pair((r.hi+r.lo)/b.hi);
 }
 Pair &operator+=(Pair b){*this=*this+b;return *this;}
 Pair &operator-=(Pair b){*this=*this-b;return *this;}
 Pair &operator*=(Pair b){*this=*this*b;return *this;}
 bool operator<(Pair b)const{return hi<b.hi || (hi==b.hi && lo<b.lo);}
 bool operator>(Pair b)const{return b<*this;}
 bool operator<=(Pair b)const{return !(b<*this);}
 bool operator>=(Pair b)const{return !(*this<b);}
 bool operator==(Pair b)const{return hi==b.hi && lo==b.lo;}
 bool operator!=(Pair b)const{return !(*this==b);}
};
using T=Pair;
auto fabsf=[](T a){return a<T(0)?-a:a;};
auto sqrtf=[](T a){
 float q=::sqrtf(a.hi);
 if(q==0)return T(0);
 T r=a-T(q)*T(q);
 return T(q)+T((r.hi+r.lo)/(2*q));
};
auto logf=[](T a){return T(::logf(a.hi))+T(a.lo/a.hi);};
auto atan2f=[](T y,T x){
 T r=T(::atan2f(y.hi,x.hi));
 T den=x*x+y*y;
 return den==T(0)?r:r+(T(x.hi)*T(y.lo)-T(y.hi)*T(x.lo))/den;
};
'''


def literal(value):
    hi=np.float32(value);lo=np.float32(value-float(hi))
    return 'T('+cpp(hi)+','+cpp(lo)+')'


def cpp(value):
    s=format(float(value),'.17g')
    if '.' not in s and 'e' not in s:s+='.0'
    return s+'f'


def elementary_header():
    """Range-reduced series on small intervals, evaluated in two floats."""
    atan='auto atan2f=[](T y,T x){\n'
    atan+='T zero=T(0),ax=x<zero?-x:x,ay=y<zero?-y:y;\n'
    atan+='if(ax==zero && ay==zero)return zero;\n'
    atan+='bool invert=ay>ax;T r=invert?ax/ay:ay/ax;\n'
    atan+=f'bool shift=r>{literal(math.sqrt(2)-1)};T t=shift?(r-T(1))/(r+T(1)):r;T z=t*t;\n'
    atan+=f'T s={literal(1/33)};\n'
    for k in range(15,-1,-1):atan+=f's=s*z+{literal((-1)**k/(2*k+1))};\n'
    atan+=f'T a=t*s+(shift?{literal(math.pi/4)}:zero);\n'
    atan+=f'if(invert)a={literal(math.pi/2)}-a;if(x<zero)a={literal(math.pi)}-a;\n'
    atan+='return y<zero?-a:a;\n};'
    log='''auto logf=[](T a){
 // This caller supplies positive normal 1+z, not a general log API.
 union Bits {float f;unsigned int u;};Bits bits;bits.f=a.hi;
 int exponent=static_cast<int>((bits.u>>23)&255)-126;
 bits.u=(bits.u&0x007fffffU)|0x3f000000U;float m=bits.f;
 T mantissa=T(m,::ldexpf(a.lo,-exponent));
'''
    log+=f'if(m<{cpp(math.sqrt(.5))}){{mantissa*=T(2);--exponent;}}\n'
    log+='T u=(mantissa-T(1))/(mantissa+T(1)),z=u*u;\n'
    log+=f'T s={literal(1/19)};\n'
    for k in range(8,-1,-1):log+=f's=s*z+{literal(1/(2*k+1))};\n'
    log+=f'return T(2)*u*s+T(exponent)*{literal(math.log(2))};\n}};'
    header=re.sub(r'auto logf=.*?\n};',lambda _:log+'\n'+atan,HEADER,flags=re.S)
    # Match spans both old elementary lambdas, ending at atan2's closing brace.
    assert '::logf' not in header and '::atan2f' not in header
    return header


def transform(code,accurate_math=False,fast=False,force_inline=False):
    header=elementary_header() if accurate_math else HEADER
    if fast:
        header=header.replace(' e+=hi*b.lo+lo*b.hi; e+=lo*b.lo;', ' e+=hi*b.lo+lo*b.hi;')
        old='''  float q=hi/b.hi;Pair r=*this-b*Pair(q);
  return Pair(q)+Pair((r.hi+r.lo)/b.hi);'''
        new='''  float q=hi/b.hi,r=::fmaf(-q,b.hi,hi);
  r+=lo;r=::fmaf(-q,b.lo,r);
  float t=r/b.hi,s=q+t;return Pair(s,t-(s-q));'''
        assert old in header
        header=header.replace(old,new)
        header=header.replace(' T r=a-T(q)*T(q);\n return T(q)+T((r.hi+r.lo)/(2*q));',
            ' float r=::fmaf(-q,q,a.hi)+a.lo;\n return T(q)+T(r/(2*q));')
    if force_inline:
        prefix='''#if defined(__CUDACC__)
#define FB_PAIR_INLINE __forceinline__
#else
#define FB_PAIR_INLINE inline
#endif
'''
        header=re.sub(r'(?m)^ (Pair\b|operator Base|bool operator)',r' FB_PAIR_INLINE \1',header)
        header=prefix+header.replace('using T=Pair;','#undef FB_PAIR_INLINE\nusing T=Pair;')
    code=code.replace('using T=float;',header)
    # Split generation-time constants, without runtime double arithmetic.
    start=code.index('T ih=')
    header,body=code[:start],code[start:]
    body=re.sub(r'T\(([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\)',lambda m:literal(float(m[1])),body)
    body=re.sub(r'T C=[^;]+;',lambda _:f'T C={literal(1365/(64*math.pi))};',body)
    body='T h_pair=T(h);\n'+re.sub(r'\bh\b','h_pair',body)
    code=header+body
    code=code.replace('auto log_one_plus=[]','auto log_one_plus=[&]').replace('auto angle=[]','auto angle=[&]')
    return code

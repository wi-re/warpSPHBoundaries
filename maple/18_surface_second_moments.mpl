# Existing compact-potential recursion extended to scalar second moments.
restart:
check_surface_moments := proc()
local r,Fx,Fy,Div,W,P,identity,l0,l1,l2,N,X,Y,px,py,K,plane,i;
r := sqrt(x^2+y^2):
Fx := x*H(r)/(x^2+y^2): Fy := y*H(r)/(x^2+y^2):
Div := simplify(diff(Fx,x)+diff(Fy,y)):
W := Div:
for P in [x^2,x*y,y^2] do
    identity := simplify(diff(P*Fx,x)+diff(P*Fy,y)-P*W-diff(P,x)*Fx-diff(P,y)*Fy):
    if identity <> 0 then error "second-moment divergence identity failed" end if:
end do:
l0 := 1-u-v: l1 := u: l2 := v:
N := [l0*(2*l0-1),l1*(2*l1-1),l2*(2*l2-1),4*l0*l1,4*l0*l2,4*l1*l2]:
X := [x0,x1,x2,(x0+x1)/2,(x0+x2)/2,(x1+x2)/2]:
Y := [y0,y1,y2,(y0+y1)/2,(y0+y2)/2,(y1+y2)/2]:
px := l0*x0+l1*x1+l2*x2: py := l0*y0+l1*y1+l2*y2:
if simplify(add(N[i]*X[i]^2,i=1..6)-px^2) <> 0 or
   simplify(add(N[i]*X[i]*Y[i],i=1..6)-px*py) <> 0 or
   simplify(add(N[i]*Y[i]^2,i=1..6)-py^2) <> 0 then
    error "P2 second-moment reproduction failed"
end if:
K := 9-84*q^2+630*q^4-1344*q^5+1260*q^6-576*q^7+105*q^8:
plane := int(q^3*K,q=0..1):
if plane <> 7/132 then error "whole-plane w4 second moment failed" end if:
printf("PASS: second-moment compact-potential divergence identities and P2 reproduction\n"):
printf("PASS: w4 whole-plane m20=m02=%a, m11=0\n",plane):
end proc:
check_surface_moments():
quit:

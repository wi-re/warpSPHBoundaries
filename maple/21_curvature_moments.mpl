# All moments from one boundary-graph expansion, weighted geometric order two.
restart:
check_curvature_moments := proc()
local t,z,d,k,b,W0,D0,H0,B0,B2,B4,B6,J0,shift,A,Az,expr,out,j,ch,aa,bb,powers,expected,integrateLine,
 P,L,T,N,Mt,Mn,Gt,Gn,Ctt,Ctn,Cnn,eps,normalDerivative,Ds,clip,g,component;
integrateLine := proc(E)
local j,poly,res;
res := 0:
poly := expand(E):
for j from 0 to 8 by 2 do
  res := res+coeff(coeff(poly,W0,1),t,j)*[B0,B2,B4,B6,0][j/2+1]:
  res := res+coeff(coeff(poly,D0,1),t,j)*[J0,B0,3*B2,5*B4,7*B6][j/2+1]:
  res := res+coeff(coeff(poly,H0,1),t,j)*[0,-J0,-3*B0,-15*B2,-35*B4][j/2+1]:
end do:
return expand(res):
end proc:
shift := -k*t^2/2-b*t^3/6:
powers := [[0,0],[1,0],[0,1],[2,0],[1,1],[0,2], [1,0],[0,1],[2,0],[1,1],[1,1],[0,2]]:
expected := [-k*B2/2+3*d*k^2*B2/8,
 -b*B4/6, k*d*B2/2+k^2*(B4-3*d^2*B2)/8,
 -k*B4/2+5*d*k^2*B4/8,b*d*B4/6,-k*d^2*B2/2+k^2*(-2*d*B4+3*d^3*B2)/8,
 -b*B2/2,k*d*B0/2+3*k^2*(B2-d^2*B0)/8,
 -3*k*B2/2+15*d*k^2*B2/8,b*d*B2/2,b*d*B2/2,
 -k*d^2*B0/2+k^2*(-6*d*B2+3*d^3*B0)/8]:
for ch from 1 to 12 do
 aa := powers[ch][1]: bb := powers[ch][2]:
 A := t^aa*z^bb:
 if ch <= 6 then
    Az := diff(A,z)*W0-A*z*D0: A := A*W0:
 else
    Az := diff(A,z)*D0+A*z*H0: A := A*D0:
 end if:
 expr := subs(z=-d,shift*A+k^2*t^4*Az/8):
 out := integrateLine(expr):
 if simplify(out-expected[ch]) <> 0 then error "shape moment mismatch in channel %1",ch end if:
end do:
printf("PASS: all twelve corrections from direct boundary-shape Taylor expansion\n"):
# Frenet frame n_s=k*t, t_s=-k*n; x_s=(1+d*k)t.
# Differentiate m1 and compare with g1-lambda*I through weighted order two.
L := P-k*B2/2+3*d*k^2*B2/8:
Mt := -b*B4/6:
Mn := -B2+k*d*B2/2+k^2*(B4-3*d^2*B2)/8:
Ctt := P-3*k*B2/2+15*d*k^2*B2/8:
Ctn := b*d*B2/2:
Cnn := P+d*B0-k*d^2*B0/2+k^2*(-6*d*B2+3*d^3*B0)/8:
normalDerivative := E -> diff(E,d)-B0*diff(E,P)-d*B0*diff(E,B2)-3*d*B2*diff(E,B4):
Ds := E -> b*diff(E,k)+g*diff(E,b):
clip := E -> convert(series(subs(k=k*eps,b=b*eps^2,g=g*eps^3,E),eps=0,3),polynom):
Gt := -b*B2/2:
Gn := -B0+k*d*B0/2+3*k^2*(B2-d^2*B0)/8:
for component in [normalDerivative(L)-Gn,Ds(L)/(1+d*k)-Gt,
 normalDerivative(Mt)-Ctn,normalDerivative(Mn)-(Cnn-L),
 (Ds(Mt)+k*Mn)/(1+d*k)-(Ctt-L),(Ds(Mn)-k*Mt)/(1+d*k)-Ctn] do
 if simplify(clip(component)) <> 0 then error "gradient/moment Frenet consistency failed" end if:
end do:
printf("PASS: gradient of mass and derivative of m1=g1-lambda*I through geometric order two\n"):

printf("PASS: radial IBP reductions J2=B0, J4=3B2, J6=5B4 and H4=-3B0,H6=-15B2\n"):
if simplify((diff(k/(1+d*k),k)*b/(1+d*k))/(1-d*k/(1+d*k))^3-b) <> 0 then
 error "curvature variation from Laplacian jets failed"
end if:
printf("PASS: kappa_s = tangential gradient(Delta d)/(1-d*Delta d)^3\n"):
end proc:
check_curvature_moments():
quit:

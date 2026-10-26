# Sphere/plane angular moments, 3D curvature invariants and small-ball terms.
restart:
check := proc(label,ok)
 if not evalb(ok) then error "FAIL: %1",label end if;
 printf("PASS: %s\n",label);
end proc:
check("A0",simplify(int(2*Pi,u=-1..c)-2*Pi*(c+1))=0):
check("A1",simplify(int(2*Pi*u,u=-1..c)-Pi*(c^2-1))=0):
check("An",simplify(int(2*Pi*u^2,u=-1..c)-2*Pi*(c^3+1)/3)=0):
check("At",simplify(int(Pi*(1-u^2),u=-1..c)-Pi*(c-c^3/3+2/3))=0):
Q2 := k1*cos(t)^2+k2*sin(t)^2:
check("curvature order one angular invariant",simplify(int(Q2,t=0..2*Pi)-Pi*(k1+k2))=0):
check("curvature order two angular invariant",simplify(int(Q2^2,t=0..2*Pi)-Pi*(3*((k1+k2)/2)^2-k1*k2))=0):
check("mass integration by parts",expand(diff((r^2-d^2)^2,r)-4*r*(r^2-d^2))=0):
check("exact sphere minus plane cap",simplify(Pi*r/(a+d)*(a^2-(a+d)^2+2*(a+d)*r-r^2)-2*Pi*r*(r-d)+Pi*r*(r^2-d^2)/(a+d))=0):
check("sphere reciprocal through curvature order two",coeff(series(1/(1/k+d),k=0,3),k,2)=-d):
# Isotropic 3D ball expectations times the Cartesian Taylor factors.
check("small-ball second Taylor coefficient",simplify((a^2/5)/2-a^2/10)=0):
check("small-ball fourth Taylor coefficient",simplify(3*(a^4/35)/factorial(4)-a^4/280)=0):
# Direct Cartesian product-Laplacian identity through degree 4, including
# negative radial powers introduced by kernel derivatives and repeated Delta.
Y := [x,y,z]: rr:=sqrt(x^2+y^2+z^2): count:=0:
for p from -6 to 11 do
 for a1 from 0 to 4 do for a2 from 0 to 4-a1 do for a3 from 0 to 4-a1-a2 do
  al:=[a1,a2,a3]: P:=x^a1*y^a2*z^a3:
  left_expr:=add(diff(P*rr^p,Y[j],Y[j]),j=1..3):
  right_expr:=p*(p+1+2*(a1+a2+a3))*P*rr^(p-2)+add(diff(P,Y[j],Y[j])*rr^p,j=1..3):
  if simplify(left_expr-right_expr) <> 0 then error "FAIL: monomial radial Laplacian",p,al end if:
  count:=count+1:
 end do:end do:end do:
end do:
check(sprintf("radial polynomial Laplacian, %d cases",count),count=630):
printf("23_radial_terms_3d: complete\n"):

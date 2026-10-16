# Conditioned triangle area coordinates and regular kernel-origin derivative.
restart:
y1 := a1+u*b1+(1-u)*t*c1:
y2 := a2+u*b2+(1-u)*t*c2:
J := expand(diff(y1,u)*diff(y2,t)-diff(y1,t)*diff(y2,u)):
if simplify(J-(1-u)*(b1*c2-b2*c1)) <> 0 then
    error "triangle Jacobian identity failed"
end if:
W := 9-84*r^2+630*r^4-1344*r^5+1260*r^6-576*r^7+105*r^8:
Q := -add(k*coeff(W,r,k)*r^(k-2),k=2..8):
if simplify(Q+diff(W,r)/r) <> 0 or limit(Q,r=0) <> 168 then
    error "regular origin derivative identity failed"
end if:
l0 := (1-u)*(1-t): l1 := u: l2 := (1-u)*t:
N := [l0*(2*l0-1),l1*(2*l1-1),l2*(2*l2-1),4*l0*l1,4*l0*l2,4*l1*l2]:
if simplify(add(N[i],i=1..6)-1) <> 0 then
    error "P2 partition of unity failed"
end if:
printf("PASS: conditioned triangle Jacobian and P2 partition of unity\n"):
printf("PASS: exact regular w4 origin derivative, -W'(r)/r -> 168\n"):
quit:

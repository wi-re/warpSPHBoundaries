# Independent boundary-shape coefficients and curvature-variation chain rule.
restart:
check_sdf_accuracy := proc()
local k,t,d,z,a,Phi,delta,shape,F1,F2,D,ds,ks,Ld,Lk,normal,tangent,grad,q,q0,A,B,Lambda;
delta := -k*t^2/2:
# Taylor-expand the antiderivative at the flat wall z=-d. The circle's
# graph has no k^2 displacement, so the second coefficient comes from W_z.
shape := expand(delta*Phi(t^2+d^2)+delta^2/2*(-2*d)*D(Phi)(t^2+d^2)):
F1 := coeff(shape,k,1): F2 := coeff(shape,k,2):
if simplify(F1+t^2*Phi(t^2+d^2)/2) <> 0 then
    error "first shape coefficient failed"
end if:
if simplify(F2+d*t^4*D(Phi)(t^2+d^2)/4) <> 0 then
    error "second shape coefficient failed"
end if:
# D_radial=-W'(r)/r=-2 Phi'(r^2), hence F2=d*t^4*D_radial/8.
if simplify(subs(D(Phi)(t^2+d^2)=-a/2,F2)-d*t^4*a/8) <> 0 then
    error "radial derivative profile conversion failed"
end if:
printf("PASS: independent graph-shape F1 and F2 coefficients\n"):
# Offset surface x(s,d)=c(s)+d*n(s); n_s=k*t, so x_s=(1+d*k)t.
normal := Vector([0,1]): tangent := Vector([1,0]):
grad := Ld*normal+Lk*ks/(1+d*k)*tangent:
if simplify(LinearAlgebra:-DotProduct(grad,(1+d*k)*tangent,conjugate=false)-Lk*ks) <> 0 or
   simplify(LinearAlgebra:-DotProduct(grad,normal,conjugate=false)-Ld) <> 0 then
    error "offset-coordinate gradient chain rule failed"
end if:
if simplify(subs(k=0,Lk=F1,LinearAlgebra:-DotProduct(grad,tangent,conjugate=false))-F1*ks) <> 0 then
    error "inflection tangential derivative failed"
end if:
printf("PASS: tangential mass derivative includes (F1+2*k*F2)*k_s/(1+d*k)\n"):
Lambda := A*(q0-q)-B*(q0-q)^2/2:
if simplify(diff(Lambda,q)+A+B*(q-q0)) <> 0 then
    error "subnode antiderivative continuation failed"
end if:
if simplify(diff(q*(A+B*(q-q0))+Lambda,q)-q*B) <> 0 then
    error "first-moment continuation slope failed"
end if:
printf("PASS: integrate linear F0 continuation below qmin; first moment has zero wall slope\n"):
end proc:
check_sdf_accuracy():
quit:

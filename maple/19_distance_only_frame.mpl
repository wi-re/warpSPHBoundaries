# Curvature-free half-plane frame: rotations and the right-angle mass limit.
restart:
check_distance_frame := proc()
local c,s,A,B,F,C,nn,M,Q,theta,identity,var;
c := cos(theta): s := sin(theta):
nn := Matrix([[c^2,c*s],[c*s,s^2]]):
M := B*LinearAlgebra:-IdentityMatrix(2)+(A-B)*nn:
Q := F*LinearAlgebra:-IdentityMatrix(2)+(C-F)*nn:
if simplify(LinearAlgebra:-Trace(M)-A-B) <> 0 or
   simplify(LinearAlgebra:-Determinant(M)-A*B) <> 0 then
    error "scalar second-moment rotation failed"
end if:
if simplify(LinearAlgebra:-Trace(Q)-F-C) <> 0 then
    error "gradient-moment rotation failed"
end if:
identity := map(simplify,subs(A=var,B=var,M)-var*LinearAlgebra:-IdentityMatrix(2)):
if not LinearAlgebra:-Equal(identity,Matrix(2,2,0)) then
    error "whole-plane isotropic moment failed"
end if:
if int(1/(2*Pi),theta=0..Pi) <> 1/2 or
   int(1/(2*Pi),theta=0..Pi/2) <> 1/4 then
    error "radial half-plane/quadrant boundary mass failed"
end if:
printf("PASS: distance-only scalar and gradient moment frame rotations\n"):
printf("PASS: half-plane vertex mass 1/2 versus quadrant mass 1/4 (geometry bias, no curvature)\n"):
end proc:
check_distance_frame():
quit:

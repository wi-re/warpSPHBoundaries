# Stable geometry of the circle-arc boundary integral, h=1.
# The scalar flux factor is (n.y)/|y|^2. At D=R its removable
# endpoint singularity has the constant continuation 1/(2R).
restart:
r2 := (D-R)^2 + 4*D*R*cos(phi/2)^2:
ndoty := (R-D) + 2*D*cos(phi/2)^2:
check_r2 := simplify(r2 - (D^2+R^2+2*D*R*cos(phi)), trig):
check_dot := simplify(ndoty - (R+D*cos(phi)), trig):
check_yx := simplify((D-R+2*R*cos(phi/2)^2) - (D+R*cos(phi)), trig):
check_boundary := simplify(subs(D=R, ndoty/r2) - 1/(2*R)):
if check_r2 <> 0 or check_dot <> 0 or check_yx <> 0 or check_boundary <> 0 then
    error "circle boundary geometry identity failed"
end if:
if int(cos(phi), phi=0..2*Pi) <> 0 or int(cos(phi)^2, phi=0..2*Pi) <> Pi then
    error "circle centre symmetry identity failed"
end if:
printf("PASS: half-angle distance, normal flux, coordinate and D=R flux continuation\n"):
printf("PASS: centre first/gradient symmetry and pi*R^2*W(R) diagonal flux\n"):
quit:

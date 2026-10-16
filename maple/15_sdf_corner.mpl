# The sharp-corner stress case: curvature recovery is singular even with an
# exact SDF. Run from the repository root: maple -q maple/15_sdf_corner.mpl
restart:
chk := proc(label, ok)
    if not ok then error "FAILED: %1", label end if;
    printf("%-72s PASS\n", label);
end proc:

# Fluid quadrant x,y>0 of the solid x<=0,y<=0: closest point is the vertex.
dc := sqrt(x^2+y^2):
lc := simplify(diff(dc,x$2)+diff(dc,y$2)):
chk("exterior corner Laplacian = 1/d", evalb(simplify(lc-1/dc)=0));
chk("exterior corner curvature denominator = 0", evalb(simplify(1-dc*lc)=0));
chk("unit gradient does NOT rule out the corner pole",
    evalb(simplify(diff(dc,x)^2+diff(dc,y)^2)=1));

# Smooth circle, in contrast: the same conversion recovers the surface 1/R.
ds := sqrt(x^2+y^2)-R:
ls := simplify(diff(ds,x$2)+diff(ds,y$2)):
chk("smooth circle recovers surface curvature 1/R",
    evalb(simplify(ls/(1-ds*ls)-1/R)=0));
print("15_sdf_corner: complete");

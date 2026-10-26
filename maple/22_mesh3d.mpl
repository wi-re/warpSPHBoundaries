# Local identities used by the analytic tetrahedron / triangle reference.
# Run with maple -q maple/22_mesh3d.mpl. No runtime exports are generated.
restart:
check := proc(label, expr)
  if not evalb(expr) then error "FAIL: %1", label end if;
  printf("PASS: %s\n", label);
end proc:

# 3D regular flux, inside/outside a truncated radial block.
for n from 0 to 14 do
  check(sprintf("3D radial divergence, n=%d", n),
        simplify(3*r^n/(n+3)+r*diff(r^n/(n+3),r)-r^n)=0);
end do:
check("exterior radial flux has zero divergence",
      simplify(3*C/r^3+r*diff(C/r^3,r))=0):

# In-plane scalar flux: rho^2=r^2-a^2 and r dr=rho d rho.
for n in [-3,-1,seq(j,j=0..16)] do
  H := (r^(n+2)-a^(n+2))/(n+2):
  check(sprintf("face radial divergence, n=%d", n),
        simplify(2*H/(r^2-a^2)+(r^2-a^2)/r*diff(H/(r^2-a^2),r)-r^n)=0);
end do:

# New elementary edge primitive and recurrence. Assume a,b>0; w enters as |w|.
r2 := sqrt(s^2+b^2+a^2):
Km1 := arctan(a*s/(b*r2))/(a*b):
check("K_-1 solid-angle-type primitive",
      simplify(diff(Km1,s)-1/(r2*(s^2+b^2)))=0):
check("K_0 circular primitive", simplify(diff(arctan(s/b)/b,s)-1/(s^2+b^2))=0):
check("K_1 base", simplify(diff(arcsinh(s/sqrt(a^2+b^2))+a^2*Km1,s)-r2/(s^2+b^2))=0)
      assuming a>0,b>0,s::real:
for n from 2 to 18 do
  check(sprintf("K_%d recurrence integrand", n),
        simplify(r2^(n-2)+a^2*r2^(n-2)/(s^2+b^2)-r2^n/(s^2+b^2))=0);
end do:
check("a=0 K_-1 base", simplify(diff(s/(b^2*sqrt(s^2+b^2)),s)-(s^2+b^2)^(-3/2))=0):

# Compact moment recursion, all |alpha|<=4, 3D. The same identity with
# y3 fixed=a is the in-plane recursion after replacing y1,y2 by u,v.
rr := sqrt(y1^2+y2^2+y3^2):
Y := [y1,y2,y3]: cnt := 0:
for n from 0 to 8 do
  Phi := (rr^(n+2)-R^(n+2))/(n+2):
  for a1 from 0 to 4 do for a2 from 0 to 4-a1 do for a3 from 0 to 4-a1-a2 do
    al := [a1,a2,a3]:
    if a1+a2+a3 = 0 then next end if:
    i := 1: while al[i]=0 do i:=i+1 end do:
    be := al: be[i] := be[i]-1:
    P := mul(Y[j]^be[j], j=1..3):
    checkval := simplify(diff(P*Phi,Y[i])-diff(P,Y[i])*Phi-mul(Y[j]^al[j],j=1..3)*rr^n):
    if checkval <> 0 then error "FAIL: moment recursion", n,al end if:
    cnt := cnt+1:
  end do: end do: end do:
end do:
check(sprintf("compact moment recursion, %d cases",cnt),cnt=306):

# P0-P3 Lagrange basis: Kronecker values and partition on triangles/tets.
for dim in [2,3] do
  for p from 1 to 3 do
    if dim=2 then
      Bs := [seq(seq([i,j,p-i-j],j=0..p-i),i=0..p)]:
    else
      Bs := [seq(seq(seq([i,j,k,p-i-j-k],k=0..p-i-j),j=0..p-i),i=0..p)]:
    end if:
    pol := []:
    for be in Bs do
      pol := [op(pol),mul(mul(p*L[j]-k,k=0..be[j]-1)/factorial(be[j]),j=1..dim+1)]:
    end do:
    check(sprintf("P%d partition, dimension %d",p,dim),
          expand(subs(L[dim+1]=1-add(L[j],j=1..dim),add(t,t in pol))-1)=0):
    for k from 1 to nops(Bs) do for j from 1 to nops(Bs) do
      checkval := subs({seq(L[t]=Bs[k][t]/p,t=1..dim+1)},pol[j]):
      if checkval <> `if`(j=k,1,0) then error "FAIL: Kronecker basis" end if:
    end do: end do:
  end do:
end do:
printf("PASS: all P1-P3 triangle/tetrahedron nodal Kronecker values\n"):
printf("22_mesh3d: complete\n"):

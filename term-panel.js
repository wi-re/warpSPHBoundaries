import {FIELD_INFO,FIELD_IDS,SHAPES} from './term-engine.js';
const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
const formulas={
 moments:'Mα = ∫Ω yα Wₕ(|y|) dV',
 gradient:'Gα,j = ∫Ω yα ∂xⱼWₕ dV = −∫Ω yα yⱼ Lₕ dV',
 hessian:'Hᵢⱼ = ∫Ω [δᵢⱼ Lₕ + yᵢyⱼ Qₕ] dV',
 laplacian:'Δλ = tr H = ∫Ω [3 Lₕ + r² Qₕ] dV',
 radial:'Lₕ = Wₕ′(r)/r,   Qₕ = Lₕ′(r)/r',
 translation:'∂xⱼMα = Gα,j − αⱼ Mα−eⱼ',
 curvature:'I[F] ≈ Iplane[F] − ½ ∫ Q₂ F dA + ⅛ ∫ Q₂² ∂zF dA,   Q₂ = κ₁u² + κ₂v²',
 curvatureMass:'λ ≈ F₀(d) − π H J(d) + ½π d (3H²−K) J(d)',
 curvatureJ:'J(d) = ∫|d|¹ r(r²−d²) W(r) dr,   H=(κ₁+κ₂)/2,   K=κ₁κ₂',
 ball:'∫Bₐ(c) F dV ≈ (4πa³/3) [F(c) + a²ΔF(c)/10 + a⁴Δ²F(c)/280]',
 cap:'c = −d/r (plane),   c = [R²−D²−r²]/(2Dr) (sphere)',
 angular:'A₀=2π(c+1),  A₁=π(c²−1),  Aₙ=2π(c³+1)/3,  Aₜ=π(c−c³/3+2/3)',
 reference:'Mα = ∑faces nᵢ Sface,β[Φ] − βᵢ Mβ−eᵢ[Φ],   α=β+eᵢ,   Φ=−∫rʰ tW(t)dt'
};
function equation(parent,text){const e=node('div',text,'term-equation');e.setAttribute('role','math');parent.append(e);}
function detail(parent,title,lines,note){const e=node('details'),summary=node('summary',title);e.append(summary);for(const line of lines)equation(e,formulas[line]??line);if(note)e.append(node('p',note,'hint'));parent.append(e);}
export function renderTermPanel(parent,c){
 const info=FIELD_INFO[FIELD_IDS[c.field]];if(!info)return;
 parent.replaceChildren();parent.append(node('h2','Equation · '+info[1]),node('p','y = source − query. Derivatives act on the kernel; Ω is the fixed solid. h is the support radius.','hint'));equation(parent,info[3]);
 const category=c.field.startsWith('g1')||c.field.startsWith('G')||c.field==='grad_norm'?'gradient':c.field.startsWith('H')?'hessian':c.field==='laplacian'?'laplacian':'moments';equation(parent,formulas[category]);
 if(category==='gradient'||category==='hessian'||category==='laplacian')equation(parent,formulas.radial);
 if(category==='gradient')equation(parent,formulas.translation);
 const kernelText={w2:'W(q) = [21/(2π)] (1−q)⁴(1+4q)',w4:'W(q) = [495/(32π)] (1−q)⁶(1+6q+35q²/3)',w6:'W(q) = [1365/(64π)] (1−q)⁸(1+8q+25q²+32q³)',cubic:'W(q) = (16/π) [(1−q)₊³ − 4(½−q)₊³]'};
 detail(parent,'Kernel and physical scaling',[kernelText[c.kernel],'Wₕ(r)=h⁻³ W(r/h),   0≤r/h≤1'],'Displayed scalar moments divide by h^|α|, gradients multiply by h, and Hessians/Laplacians multiply by h². g₁ needs no scaling. Raw exports use these same dimensionless quantities.');
 detail(parent,'Sphere / local-plane radial integrals',['cap','angular','M₀=∫r²W A₀dr;  M₁,n=∫r³W A₁dr;  M₂,n/t=∫r⁴W Aₙ/ₜdr','Gₙ=−∫r³L A₁dr;  g₁,n/t=−∫r⁴L Aₙ/ₜdr','Hₙ/ₜ=∫[r²L A₀+r⁴Q Aₙ/ₜ]dr'],'Clamp c to [−1,1]; split radial intervals at complete/partial-shell boundaries and kernel knots. Analytic derivative integrands replace finite differences of the mass grid.');
 detail(parent,'Curvature terms through order two',['curvature','curvatureMass','curvatureJ'],'Here z=−d at the tangent plane; Ω lies below the quadratic graph z=−d−Q₂/2. Positive κ is convex outward curvature. The viewer uses the umbilic sphere κ₁=κ₂=1/R, with R/h≥2. The domain cutoff is not an integral-error guarantee; higher graph/variation terms remain. Every displayed channel expands its own integrand F.');
 detail(parent,'Thin-volume / small-ball terms',['ball','Δ(yαrᵖ) = p(p+1+2|α|)yαrᵖ⁻² + ∑ᵢ αᵢ(αᵢ−1)yα−2eᵢrᵖ'],'Apply the mean-value series to the selected integrand, including derivatives. The viewer requires a/h≤0.3, excludes the radial origin, and rejects support/kernel-knot crossings. Missing samples stay missing.');
 detail(parent,'Tetrahedron and triangular-surface reference',['reference','∫T W dV = ∑faces zface ∫face M₃(r)/r³ dA,   M₃(r)=∫₀ʳ t²W(t)dt'],'The Python arbitrary-precision mesh3d reference supplies analytic face-to-edge checks. Browser solid cubature preserves the actual preset geometry but approximates its integral; it is not that exact reference implementation.');
 const note=node('p','Precision: browser JavaScript uses float64 arithmetic; WebGPU compute uses float32; display textures use float16. Cubature error, local-model error, asymptotic truncation and arithmetic error are separate.','hint');parent.append(note);parent.append(node('p',`Selected expansions: curvature order ${c.curvature_order??2}; small-ball order ${c.series_order??4}.`,'hint'));
}

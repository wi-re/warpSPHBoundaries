"""independent reference: Stokes flow through a periodic square array of cylinders by volume-penalised Fourier-CG; drag per length K = F/(mu U), U the superficial velocity."""
import numpy as np, math, sys
def K_of(c, N=512, eta=1e6, tol=1e-10):
    L=1.0; R=math.sqrt(c/math.pi)
    x=(np.arange(N)+0.5)/N
    X,Y=np.meshgrid(x,x,indexing="ij")
    r=np.sqrt((X-0.5)**2+(Y-0.5)**2)
    w=1.5/N
    chi=0.5*(1-np.tanh((r-R)/(0.35*w)))     # smoothed solid indicator
    k=2*np.pi*np.fft.fftfreq(N,1.0/N)
    KX,KY=np.meshgrid(k,k,indexing="ij"); K2l=KX**2+KY**2; K2=K2l.copy(); K2[0,0]=1.0
    nu=1.0; f=np.array([1.0,0.0])
    def proj(ux,uy):
        fx,fy=np.fft.fft2(ux),np.fft.fft2(uy)
        d=(KX*fx+KY*fy)/K2
        fx-=KX*d; fy-=KY*d
        return np.real(np.fft.ifft2(fx)),np.real(np.fft.ifft2(fy))
    def A(u):
        ux,uy=u
        lx=np.real(np.fft.ifft2(K2l*np.fft.fft2(ux))); ly=np.real(np.fft.ifft2(K2l*np.fft.fft2(uy)))
        lx[...]=lx; 
        ax=nu*lx+eta*chi*ux; ay=nu*ly+eta*chi*uy
        return proj(ax,ay)
    def prec(u):
        ux,uy=u
        s=1.0/(nu*K2l+eta*chi.mean()+1e-30)
        return np.real(np.fft.ifft2(s*np.fft.fft2(ux))),np.real(np.fft.ifft2(s*np.fft.fft2(uy)))
    # solve A u = proj(f) with u divergence-free incl. mean
    b=proj(np.full((N,N),f[0]),np.full((N,N),f[1]))
    u=(np.zeros((N,N)),np.zeros((N,N)))
    r_=b; z=prec(r_); p=z; rz=sum((a*bb).sum() for a,bb in zip(r_,z)); b2=sum((a*a).sum() for a in b)
    for it in range(30000):
        Ap=A(p); a=rz/sum((x_*y_).sum() for x_,y_ in zip(p,Ap))
        u=(u[0]+a*p[0],u[1]+a*p[1]); r_=(r_[0]-a*Ap[0],r_[1]-a*Ap[1])
        if sum((x_*x_).sum() for x_ in r_)<tol**2*b2: break
        z=prec(r_); rz2=sum((x_*y_).sum() for x_,y_ in zip(r_,z)); p=(z[0]+rz2/rz*p[0],z[1]+rz2/rz*p[1]); rz=rz2
    U=u[0].mean()                     # superficial velocity (u ~ 0 in the solid)
    F=(eta*chi*u[0]).mean()           # force on the solid per cell (unit length)
    return 1.0/(nu*U), F/f[0], it
if __name__=="__main__":
    for c in (0.1257,):
        for N in (128,256,384):
            print(c,N,K_of(c,N))

# 2D-Air-Water-Couette-Flow-CFD-Simulation-Python-NumPy-
This project simulates steady, laminar airflow over a flat water surface using the 2D incompressible Navier–Stokes equations with an energy equation. It models an air layer sheared by a moving top lid and heated from above. The simulation computes velocity, pressure, and temperature fields using a finite-difference projection method. 
# 2D Air–Water Couette Flow CFD Simulation

Simulates laminar air flow over a stationary water surface using the **incompressible Navier–Stokes equations** with an **energy equation**.

 Features
- Lid-driven air flow over a flat interface
- Temperature and velocity coupling
- Contour and streamline plots
- Profiles of velocity and temperature vs height

Physics
- Incompressible Navier–Stokes (projection method)
- Constant properties, laminar regime
- Energy equation for thermal diffusion
- No-slip walls, fixed top/bottom temperatures

2D Air–Water Couette Flow CFD Simulation
----------------------------------------
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

def build_up_b(u, v, dx, dy, dt, rho):
    b = np.zeros_like(u)
    b[1:-1,1:-1] = rho * (
        1/dt*((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx) + (v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))
        - ((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx))**2
        - 2*((u[2:,1:-1]-u[0:-2,1:-1])/(2*dy) * (v[1:-1,2:]-v[1:-1,0:-2])/(2*dx))
        - ((v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))**2
    )
    return b

def pressure_poisson(p, b, dx, dy, max_iter=200, tol=1e-6):
    pn = p.copy()
    for _ in range(max_iter):
        pn[:] = p
        p[1:-1,1:-1] = (
            ((pn[1:-1,2:]+pn[1:-1,0:-2])*dy**2 + (pn[2:,1:-1]+pn[0:-2,1:-1])*dx**2)
            / (2*(dx**2+dy**2))
            - dx**2*dy**2/(2*(dx**2+dy**2))*b[1:-1,1:-1]
        )
        # BCs
        p[:,-1] = p[:,-2]
        p[:,0]  = p[:,1]
        p[-1,:] = 0
        p[0,:]  = p[1,:]
        if np.linalg.norm(p - pn) < tol:
            break
    return p

def run_sim(nx=81, ny=21, Lx=0.1, Ly=0.01, U_lid=1.0,
            rho=1.225, mu=1.8e-5, T_water=293.15, T_lid=298.15,
            dt=1e-4, nt=3000, plot=True):
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    nu = mu/rho
    k_air, cp_air = 0.025, 1005.0
    alpha = k_air/(rho*cp_air)

    u, v, p = np.zeros((ny,nx)), np.zeros((ny,nx)), np.zeros((ny,nx))
    T = np.ones((ny,nx))*((T_water+T_lid)/2)
    x, y = np.linspace(0,Lx,nx), np.linspace(0,Ly,ny)
    X, Y = np.meshgrid(x, y)

    for n in range(nt):
        un, vn, Tn = u.copy(), v.copy(), T.copy()
        b = build_up_b(un, vn, dx, dy, dt, rho)
        p = pressure_poisson(p, b, dx, dy)

        u[1:-1,1:-1] = (un[1:-1,1:-1]
            - dt*( un[1:-1,1:-1]*(un[1:-1,1:-1]-un[1:-1,0:-2])/dx
                  + vn[1:-1,1:-1]*(un[1:-1,1:-1]-un[0:-2,1:-1])/dy )
            - dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,0:-2])
            + nu*dt*((un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])/dx**2
                     +(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1])/dy**2)
        )

        v[1:-1,1:-1] = (vn[1:-1,1:-1]
            - dt*( un[1:-1,1:-1]*(vn[1:-1,1:-1]-vn[1:-1,0:-2])/dx
                  + vn[1:-1,1:-1]*(vn[1:-1,1:-1]-vn[0:-2,1:-1])/dy )
            - dt/(2*rho*dy)*(p[2:,1:-1]-p[0:-2,1:-1])
            + nu*dt*((vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])/dx**2
                     +(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])/dy**2)
        )

        T[1:-1,1:-1] = (Tn[1:-1,1:-1]
            - dt*( un[1:-1,1:-1]*(Tn[1:-1,1:-1]-Tn[1:-1,0:-2])/dx
                  + vn[1:-1,1:-1]*(Tn[1:-1,1:-1]-Tn[0:-2,1:-1])/dy )
            + alpha*dt*((Tn[1:-1,2:]-2*Tn[1:-1,1:-1]+Tn[1:-1,0:-2])/dx**2
                        +(Tn[2:,1:-1]-2*Tn[1:-1,1:-1]+Tn[0:-2,1:-1])/dy**2)
        )

        # Boundary conditions
        u[0,:], v[0,:] = 0, 0
        u[-1,:], v[-1,:] = U_lid, 0
        u[:,0], u[:,-1] = u[:,1], u[:,-2]
        v[:,0], v[:,-1] = v[:,1], v[:,-2]
        T[0,:], T[-1,:] = T_water, T_lid
        T[:,0], T[:,-1] = T[:,1], T[:,-2]

        if n % 200 == 0:
            print(f"Step {n}/{nt}")

    # interface properties
    dy = y[1]-y[0]
    tau = mu*(u[1,:]-u[0,:])/dy
    q_flux = -k_air*(T[1,:]-T[0,:])/dy
    print("Mean τ (Pa):", np.mean(tau))
    print("Mean q'' (W/m²):", np.mean(q_flux))

    # --- PLOTS ---
    if plot:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.contourf(X,Y,u,20); plt.colorbar(); plt.title("Velocity u [m/s]")
        plt.subplot(1,3,2)
        plt.contourf(X,Y,T,20); plt.colorbar(); plt.title("Temperature [K]")
        plt.subplot(1,3,3)
        plt.streamplot(X,Y,u,v,color='k',density=1)
        plt.title("Streamlines")
        plt.tight_layout()
        plt.show()

        # centerline profiles
        u_center = u[:, nx//2]
        T_center = T[:, nx//2]
        plt.figure(figsize=(6,4))
        plt.plot(u_center, y, label="Velocity profile (u)")
        plt.plot((T_center - T_water)/(T_lid - T_water), y, label="Normalized Temp profile")
        plt.xlabel("u [m/s] or normalized T")
        plt.ylabel("Height y [m]")
        plt.title("Profiles across air layer")
        plt.legend(); plt.grid(True); plt.show()

    return dict(x=x, y=y, u=u, v=v, T=T, tau=tau, q_flux=q_flux)
    <img width="1200" height="400" alt="Figure_1oooooooooooooooo" src="https://github.com/user-attachments/assets/24fa1973-6b52-463f-9615-37cc6dcda73e" />
    <img width="600" height="400" alt="Figure_1 pppppp" src="https://github.com/user-attachments/assets/cc2598d4-b7e9-43ec-817d-53b64e5a4f7d" />



if __name__ == "__main__":
    run_sim()

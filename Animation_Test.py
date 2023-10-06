import numpy as np
import numpy.linalg as npl
import math
import matplotlib.pyplot as plt
import matplotlib.animation as mathimation
import matplotlib as mat
from matplotlib.gridspec import GridSpec
import pathlib
import scipy


t0 = 0
tShift = 0*3600
G = 6.67e-20
scale = 10000

# Satellite 1
mass = 1350
cD = 2
area = 3.6
rho0 = 4e-13
r0 = 7298.145
alt0 = 200
theta_dot = 7.2921158553e-5

# Body 1
mass1 = 5.97219e24
#mass1 = 7.348e22
R1 = 6378.1
mu1 = G * (mass1)

# Body 2
mass2 = 7.34767309e22
R2 = 1740
mu2 = G * (mass2)
ICs_body2 = np.radians([90, 0])
#ICs_body2 = np.radians([90, 10])
mean_orbit = 384400 + R1 + R2
L = [b * mean_orbit for b in [np.cos(ICs_body2[1])*np.cos(ICs_body2[0]), np.cos(ICs_body2[1])*np.sin(ICs_body2[0]), np.sin(ICs_body2[1])]]

#ICs = [7272.2, 0, 0, 0, .921 ,0]
#ICs = [2137.3, 0, 0, 0, 1.7108, 0]
#ICs = [-1264.61, 8013.81, -3371.25, -6.03962, -.204398, 2.09672]
#ICs = [8000, .1, 30, 145, 120, 10]
#ICs = [-8000, 1.1, 30, 145, 120, 10]
ICs = [-2436.45, -2436.45, 6891.0379, 5.088611, -5.088611, 0]
#ICs = [-6796, 4025, 3490, -3.7817, -6.0146, 1.1418]
#ICs = [9056, .142, 7.2, 200, 60, 320]
#ICs = [9845.01, .1508, .4363, 1.361, .5619, .7123]
#ICs = [18877, 27406.6, -19212.8, 3.55968, 6.35532, -4.18447]

tol = 1e-13

#direction = input("Enter 1 if RV was given or 2 if OE was given: ")
#tshift = input("Enter the time shift in seconds:  ")
direction = 1

t_values = []
x, y, z, vx, vy, vz, r = [], [], [], [], [], [], []
KE, PE, totalE, totalEnergyError = [], [], [], []
H, Hmag, angularMomentumError = [], [], []
xdata, ydata, zdata = [], [], []
xdata2, ydata2, deltaE = [], [], []
inclination, a, e, n, omegac, omegal, trueAnomaly = [], [], [], [], [], [], []
temp = []

def f(E, e, Me):
    return E-e*np.sin(E)-Me

def df(E, e): 
    return 1-e*np.cos(E)

def eKeplerSolution(f, df, E0, tol, e, Me):
    if df(E0, e) < tol:
        return E0
    elif np.abs(f(E0, e, Me)/df(E0, e)) < tol:
        return E0
    else:
        return eKeplerSolution(f, df, E0-f(E0, e, Me)/df(E0, e), tol, e, Me)

def f(E, e, He):
    return He-e*np.sinh(E)+E

def df(E, e): 
    return e*np.cosh(E)-1

def hKeplerSolution(f, df, E0, tol, e, He):
    if df(E0, e) < tol:
        return E0
    elif np.abs(f(E0, e, He)/df(E0, e)) < tol:
        return E0
    else:
        return hKeplerSolution(f, df, E0-f(E0, e, He)/df(E0, e), tol, e, He)

def rho(RV):
    result = rho0 * math.exp(-(npl.norm([RV[0], RV[1], RV[2]])-r0)/alt0)

    return result

def dR_2body(time,RV):

    x = RV[0]
    y = RV[1]
    z = RV[2]
    vx = RV[3]
    vy = RV[4]
    vz = RV[5]
    r1 = np.sqrt(x**2+y**2+z**2)
    correctedRho = rho(RV)
    
    xsto2 = L[0] - x
    ysto2 = L[1] - y
    zsto2 = L[2] - z
    
    rsto2 = np.sqrt(xsto2**2 + ysto2**2 + zsto2**2)
    r2 = np.sqrt(L[0]**2 + L[1]**2 + L[2]**2)

    vrel = [vx+theta_dot*y, vy-theta_dot*x, vz]

    ax = -mu1*x/r1**3# + mu2*((xsto2/rsto2**3)-(L[0]/r2**3)) - (.5*cD*area*correctedRho*vrel[0]**2)/mass
    ay = -mu1*y/r1**3# + mu2*((ysto2/rsto2**3)-(L[1]/r2**3)) - (.5*cD*area*correctedRho*vrel[1]**2)/mass
    az = -mu1*z/r1**3# + mu2*((zsto2/rsto2**3)-(L[2]/r2**3)) - (.5*cD*area*correctedRho*vrel[2]**2)/mass

    result = [vx,vy,vz,ax,ay,az]

    return result

def rotate(axis, angle):
    if axis == 1:
        R = [[1, 0, 0], [0, np.cos(angle), np.sin(angle)], [0, -np.sin(angle), np.cos(angle)]]
    elif axis == 2:
        R = [[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]]
    else:
        R = [[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    return R


def convert(input, direction):

    if direction == 1:
        x = input[0]
        y = input[1]
        z = input[2]
        vx = input[3]
        vy = input[4]
        vz = input[5]

        h = np.cross([x, y, z], [vx, vy, vz])
        r = np.sqrt(x**2 + y**2 + z**2)
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        n = np.cross([0, 0, 1], h)
        epsilon = (v**2)/2 - mu1/r

        a = -mu1/(2*epsilon)
        e = (1/mu1)*np.cross([vx, vy, vz], h) - [x, y, z]/r
        i = (np.arccos(np.dot(h, [0,0,1])/npl.norm(h)))
        omegac = (np.arctan2(np.dot(n, [0,1,0]), np.dot(n, [1,0,0])))
        omegal = (np.arctan2(np.dot(e, np.cross(h, n)), np.dot(e, n)*npl.norm(h)))
        trueAnomaly = (np.arctan2(np.dot([x, y, z], np.cross(h, e)), np.dot([x, y, z], e)*npl.norm(h)))

        result = [a, e, i, omegac, omegal, trueAnomaly]

        return result

    else:
        a = input[0]
        e = npl.norm(input[1])
        i = input[2]
        omegac = input[3]
        omegal = input[4]
        trueAnomaly = input[5]

        P = a*(1-e**2)

        xpqw = (P*np.cos(trueAnomaly))/(1+e*np.cos(trueAnomaly))
        ypqw = (P*np.sin(trueAnomaly))/(1+e*np.cos(trueAnomaly))
        zpqw = 0
        vxpqw = -np.sqrt(mu1/P)*np.sin(trueAnomaly)
        vypqw = np.sqrt(mu1/P)*(e + np.cos(trueAnomaly))
        vzpqw = 0

        rpqw = [xpqw, ypqw, zpqw]
        vpqw = [vxpqw, vypqw, vzpqw]

        reci = np.dot(np.dot(np.dot(np.transpose(rotate(3, omegac)), np.transpose(rotate(1, i))), np.transpose(rotate(3, omegal))), np.transpose(rpqw))
        veci = np.dot(np.dot(np.dot(np.transpose(rotate(3, omegac)), np.transpose(rotate(1, i))), np.transpose(rotate(3, omegal))), np.transpose(vpqw))

        reci = np.transpose(reci)
        veci = np.transpose(veci)
        result = np.concatenate((reci, veci), axis=None)

        return result

if direction == 2:
    t = tShift +t0
    a2 = ICs[0]
    e2 = npl.norm(ICs[1])
    ICs[2] = np.radians(ICs[2])
    ICs[3] = np.radians(ICs[3])
    ICs[4] = np.radians(ICs[4])
    ICs[5] = np.radians(ICs[5])
    if e2 > 1:
        H2 = np.arcsinh((np.sqrt(e2**2-1)*np.sin(ICs[5]))/(1+e2*np.cos(ICs[5])))
        He = e2*np.sinh(H2) - H2 - np.sqrt(mu1/(-(a2**3)))*t
        E0 = H2
        H1 = hKeplerSolution(f, df, E0, tol, e2, He)
        ICs[5] = 2*np.arctan(np.sqrt((e2+1)/(e2-1))*np.tanh(H1/2))

    elif e2 < 1 and e2 >= 0:
        Me2 = 2*np.arctan(np.sqrt((1-e2)/(1+e2))*np.tan(ICs[5]/2)) - ((e2*np.sqrt(1-e2**2)*np.sin(ICs[5]))/(1+e2*np.cos(ICs[5])))
        Me1 = Me2 - np.sqrt(mu1/a2**3)*t
        E0 = Me1 + e2*np.sin(Me1)
        E = eKeplerSolution(f, df, E0, tol, e2, Me1)
        ICs[5] = np.arctan2((np.sqrt(1-e2**2)*np.sin(E))/(1-e2*np.cos(E)), (e2-np.cos(E))/(e2*np.cos(E)-1))
   
    ICs = convert(ICs, 2)

else:
    t = tShift + t0
    ICs = convert(ICs, 1)
    a2 = ICs[0]
    e2 = npl.norm(ICs[1])
    if e2 > 1:
        H2 = np.arcsinh((np.sqrt(e2**2-1)*np.sin(ICs[5]))/(1+e2*np.cos(ICs[5])))
        He = e2*np.sinh(H2) - H2 - np.sqrt(mu1/(-(a2**3)))*t
        E0 = H2
        H1 = hKeplerSolution(f, df, E0, tol, e2, He)
        ICs[5] = 2*np.arctan(np.sqrt((e2+1)/(e2-1))*np.tanh(H1/2))

    elif e2 < 1 and e2 >= 0:
        Me2 = 2*np.arctan(np.sqrt((1-e2)/(1+e2))*np.tan(ICs[5]/2)) - ((e2*np.sqrt(1-e2**2)*np.sin(ICs[5]))/(1+e2*np.cos(ICs[5])))
        Me1 = Me2 - np.sqrt(mu1/a2**3)*t
        E0 = Me1
        E = eKeplerSolution(f, df, E0, tol, e2, Me1)
        ICs[5] = np.arctan2((np.sqrt(1-e2**2)*np.sin(E))/(1-e2*np.cos(E)), (e2-np.cos(E))/(e2*np.cos(E)-1))
   
    ICs = convert(ICs, 2)


#T = 2*np.pi*np.sqrt(a2**3/mu1)
nOfOrbits = 10
tf = 20*3600
#tf = nOfOrbits*T
dt = 20
fps = 120
n = tf/dt
time = np.arange(t0, tf+dt, dt)
timeSize = len(time)
frameArry = np.arange(0,timeSize,1)
frames = len(frameArry)

print(ICs)

tRV = scipy.integrate.RK45(dR_2body, t0, ICs, np.max(time), rtol=tol, atol=tol, first_step=dt, max_step=dt)

for i in time:
    tRV.step()
    t_values.append(tRV.t)
    x.append(tRV.y[0])
    y.append(tRV.y[1])
    z.append(tRV.y[2])
    vx.append(tRV.y[3])
    vy.append(tRV.y[4])
    vz.append(tRV.y[5])
    if tRV.status == 'finished':
        break

## Energy Calculations
KE0 = (vx[0]**2+vy[0]**2+vz[0]**2)/2
PE0 = -mu1/np.sqrt(x[0]**2+y[0]**2+z[0]**2)
totalE0 = KE0+PE0

for i in np.arange(0, len(x), 1):
    KE.append((vx[i]**2+vy[i]**2+vz[i]**2)/2)
    PE.append(-mu1/np.sqrt(x[i]**2+y[i]**2+z[i]**2))
    totalE.append(KE[i]+PE[i])
    totalEnergyError.append(np.abs(totalE[i]-totalE0)/totalE0)
    r.append(np.sqrt(x[i]**2+y[i]**2+z[i]**2))
    deltaE.append(totalE[i]-totalE0)

for i in np.arange(0, len(x), 1):
    H.append(np.cross([x[i],y[i],z[i]],[vx[i],vy[i],vz[i]]))
    temp.append(convert([x[i], y[i], z[i], vx[i], vy[i], vz[i]], 1))
    temp[i][1] = npl.norm(temp[i][1])
    a.append(temp[i][0])
    e.append(temp[i][1])
    inclination.append(temp[i][2])
    omegac.append(temp[i][3])
    omegal.append(temp[i][4])
    trueAnomaly.append((temp[i][5]+np.pi)*(180/np.pi))



H0mag = np.linalg.norm(np.cross([x[0],y[0],z[0]],[vx[0],vy[0],vz[0]]))

for i in np.arange(0, len(x), 1):
    Hmag.append(np.sqrt(H[i][0]**2+H[i][1]**2+H[i][2]**2))
    angularMomentumError.append(np.abs(Hmag[i]-H0mag)/H0mag)



## Define video compiler
metadata = dict(title='Sine Animation Test', artist='Charles', comment='Particle following sine wave path')
videoWriter = mathimation.FFMpegWriter(fps=fps, metadata=metadata)
mat.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'

## Initialize Plots
fig = plt.figure(figsize=(16,9), layout="constrained")
gs = GridSpec(4, 6, figure=fig)
ax = fig.add_subplot(gs[2:, 4:6], projection='3d')
ax.set_axis_off()
ax.dist=9
particle_path, = plt.plot(x, y, z, 'b', zorder=0, linewidth=.5)
planet = plt.plot(0, 0, 0, color='blue', marker="o", zorder=1)
moon = plt.plot(L[0], L[1], L[2], color='grey', marker="o", zorder=1)

#Specific energy subplot
ax2 = fig.add_subplot(gs[:-2, 2:4])
ax11 = ax2.twinx()
ax2.plot(frameArry,KE, label='Kinetic Energy')
ax2.plot(frameArry,PE, label='Potential Energy')
ax2.plot(frameArry,totalE, label='Total Energy')
ax11.plot(frameArry, deltaE, label='Change in Energy', color='red')

#Error subplot
ax3 = fig.add_subplot(gs[:-3, :-4])
ax3.plot(frameArry,totalEnergyError, label='Total Energy Error')
ax3.plot(frameArry,angularMomentumError, label='Angular Momentum Error')

#Position subplot
ax4 = fig.add_subplot(gs[:-2, 4:])
ax4.plot(frameArry,r, label='Position Magnitude')

#Angles subplots
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(frameArry,inclination, label='Inclination Angle')
ax6 = fig.add_subplot(gs[3, 0])
ax6.plot(frameArry,omegac, label='Node')
ax7 = fig.add_subplot(gs[2, 1])
ax7.plot(frameArry,omegal, label='Aergument of Perigee')
ax8 = fig.add_subplot(gs[3, 1])
ax8.plot(frameArry,trueAnomaly, label='True Anomaly')

#Characteristics subplots
ax9 = fig.add_subplot(gs[1, 0])
ax9.plot(frameArry,a, label='a')
ax10 = fig.add_subplot(gs[1, 1])
ax10.plot(frameArry,e, label='e')

#2D Projection
ax12 = fig.add_subplot(gs[2:, 2:4])
ax12.plot(x,y, label='2D Projection')
ax12.plot(0, 0, color='blue', marker='o')
#ax12.plot(L[0], L[1], color='grey', marker='o')

#Orbit planet sweep
ln, = ax.plot([], [], [], 'ro')

#Position point sweep
ln2, = ax4.plot([], [], 'ro')

#2D Orbit Projection sweep
ln3, = ax12.plot([], [], 'ro')

#Vertical line sweep
vl1 = ax2.axvline(0, ls='-', color='r', lw=1, zorder=10)
ax2.set_xlim(0,np.max(time))
vl2 = ax3.axvline(0, ls='-', color='r', lw=1, zorder=11)
ax3.set_xlim(0,np.max(time))
vl3 = ax5.axvline(0, ls='-', color='r', lw=1, zorder=11)
ax5.set_xlim(0,np.max(time))
vl4 = ax6.axvline(0, ls='-', color='r', lw=1, zorder=11)
ax6.set_xlim(0,np.max(time))
vl5 = ax7.axvline(0, ls='-', color='r', lw=1, zorder=11)
ax7.set_xlim(0,np.max(time))
vl6 = ax8.axvline(0, ls='-', color='r', lw=1, zorder=11)
ax8.set_xlim(0,np.max(time))
patches = [ln, vl1, vl2, ln2, vl3, vl4, vl5, vl6, ln3]

def init():
    ax2.set_xlim(0, frames)
    ax2.set_xlabel("Time Step (Time/dt)")
    ax3.set_xlim(0, frames)
    ax3.set_xlabel("Time Step (Time/dt)")
    ax4.set_xlim(0, frames)
    ax4.set_xlabel("Time Step (Time/dt)")
    ax5.set_xlim(0, frames)
    ax5.set_xlabel("Time Step (Time/dt)")
    ax6.set_xlim(0, frames)
    ax6.set_xlabel("Time Step (Time/dt)")
    ax7.set_xlim(0, frames)
    ax7.set_xlabel("Time Step (Time/dt)")
    ax8.set_xlim(0, frames)
    ax8.set_xlabel("Time Step (Time/dt)")
    ax9.set_xlim(0, frames)
    ax9.set_xlabel("Time Step (Time/dt)")
    ax10.set_xlim(0, frames)
    ax10.set_xlabel("Time Step (Time/dt)")
    ax12.set_xlabel("x (X-Y Projection)")
    if (np.min(totalEnergyError) > np.min(angularMomentumError)):
        if (np.max(totalEnergyError) > np.max(angularMomentumError)):
            ax3.set_ylim(np.min(angularMomentumError), np.max(totalEnergyError))
        elif (np.max(totalEnergyError) < np.max(angularMomentumError)):
            ax3.set_ylim(np.min(angularMomentumError), np.max(angularMomentumError))
    elif (np.min(totalEnergyError) < np.min(angularMomentumError)):
        if (np.max(totalEnergyError) > np.max(angularMomentumError)):
            ax3.set_ylim(np.min(totalEnergyError), np.max(totalEnergyError))
        elif (np.max(totalEnergyError) < np.max(angularMomentumError)):
            ax3.set_ylim(np.min(totalEnergyError), np.max(angularMomentumError))
    if (np.min(KE) > np.min(PE)):
        if (np.max(KE) > np.max(PE)):
            ax2.set_ylim(np.min(PE)+.1*np.min(PE), np.max(KE)+.1*np.max(KE))
        elif (np.max(KE) < np.max(PE)):
            ax2.set_ylim(np.min(PE)+.1*np.min(KE), np.max(PE)+.1*np.max(PE))
    elif (np.min(KE) < np.min(PE)):
        if (np.max(KE) > np.max(PE)):
            ax2.set_ylim(np.min(KE)+.1*np.min(KE), np.max(KE)+.1*np.max(KE))
        elif (np.max(KE) < np.max(PE)):
            ax2.set_ylim(np.min(KE)+.1*np.min(KE), np.max(PE)+.1*np.max(PE))
    ax4.set_ylim(np.min(r), np.max(r))
    ax2.set_ylabel("\u03b5 ($km^2$/$s^2$)")
    ax3.set_ylabel("% Error")
    ax4.set_ylabel("Position Magnitude (km)")
    ax5.set_ylabel("i (Rad)")
    ax5.set_ylim(-np.pi, np.pi)
    ax6.set_ylabel("\u03a9 (Rad)")
    ax6.set_ylim(-np.pi, np.pi)
    ax7.set_ylabel("\u03c9 (Rad)")
    ax7.set_ylim(-np.pi, np.pi)
    ax8.set_ylabel("\u03bd (Rad)")
    ax8.set_ylim(0, 360)
    ax9.set_ylabel("a (km)")
    ax9.set_ylim(np.min(a)-.2*np.abs(np.min(a)), np.max(a)+np.max(a))
    ax10.set_ylabel("e")
    ax10.set_ylim(np.min(e)-.2*np.abs(np.min(e)), np.max(e)+np.max(e))
    ax11.set_ylabel(u'Δ\u03b5\u209C\u2092\u209C')
    ax12.set_ylabel('y (X-Y Projection)')
    ln2.set_data(frameArry, r)
    plt.figtext(.9,.06, "Time Scale: x%.1f" % (dt+fps), ha="center", fontsize=7, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    ax2.legend(loc='lower right')
    ax3.legend(loc='lower right')
    ax4.legend(loc='lower right')
    return patches

def get_arrow(frame):
    a_x = x[frame]
    a_y = y[frame]
    a_z = z[frame]
    a_vx = scale*vx[frame]
    a_vy = scale*vy[frame]
    a_vz = scale*vz[frame]
    return a_x,a_y,a_z,a_vx,a_vy,a_vz

#quiver = ax.quiver(*get_arrow(0))

def update(frame):
    #global quiver
    #quiver.remove()
    #quiver = ax.quiver(*get_arrow(frame))
    xdata.clear()
    ydata.clear()
    zdata.clear()
    xdata2.clear()
    ydata2.clear()
    xdata.append(x[frame])
    ydata.append(y[frame])
    zdata.append(z[frame])
    xdata2.append(frame)
    ydata2.append(r[frame])
    #plt.figtext(.8,.07, u"Inclination Angle = %.4f\N{DEGREE SIGN}\n RAAN = %.2f\N{DEGREE SIGN}\n Argument of Periapsis = %.2f\N{DEGREE SIGN}" % (inclination[frame], omegac[frame], omegal[frame]), ha="center", fontsize=7, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    ln.set_data(xdata, ydata)
    ln.set_3d_properties(zdata)
    vl1.set_xdata([frame,frame])
    vl2.set_xdata([frame,frame])
    vl3.set_xdata([frame,frame])
    vl4.set_xdata([frame,frame])
    vl5.set_xdata([frame,frame])
    vl6.set_xdata([frame,frame])
    ln2.set_data(xdata2, ydata2)
    ln3.set_data(xdata, ydata)
    print(frame)
    return patches


ani = mathimation.FuncAnimation(fig, update, frames=frames, init_func=init, interval=1, blit=True)

plt.show()
ani.save('animation.mp4', writer=videoWriter)

plt.savefig('Figure1.png')

plt.close()
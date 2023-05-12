import numpy as np 
import math as Math
import scipy.optimize as optimize 
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Vi definierar våran prisfunktion för vardera klass
def pris(antal,totplats,nummer):

    pris = 600-antal/totplats*400
    if nummer == 1:
        pris = 1000-600*antal/totplats
    if nummer == 2:
        pris = 3000-2200*antal/totplats
    return pris

#Vi definierar våran kostnadsfunktion för vardera klass
def kostnad(antal, totplats, nummer):
    kostnad =  antal*50 
    if nummer == 1:
        kostnad = antal*100
    if nummer == 2:
        kostnad = antal*250 
    return kostnad

#Vi definierar våran omsättningsfunktion för vardera klass
def omsattning(antal, totplats, nummer):
    p = pris(antal, totplats, nummer)
    omsattning = p*antal
    return omsattning

#Vi definierar våran vinstfunktion för vardera klass
def vinst(antal, totplats, nummer):
    oms = omsattning(antal, totplats, nummer)
    kos = kostnad(antal, totplats, nummer)
    vinst = oms - kos
    return vinst

#Våra antal passagerar för vardera klass
antal_ak = np.arange(0,1001)
antal_tk = np.arange(0,1001)
antal_fk = np.arange(0,1001)

#Våra listor för att spara alla värden
ak_kostnader = []
tk_kostnader = []
fk_kostnader = []
ak_omsattningar = []
tk_omsattningar = []
fk_omsattningar = []
ak_vinster = []
tk_vinster = []
fk_vinster = []

#Beräkning av alla värden
for i in range(len(antal_ak)):
    k1 = kostnad(antal_ak[i],antal_ak.max(),0)
    k2 = kostnad(antal_ak[i],antal_ak.max(),1)
    k3 = kostnad(antal_ak[i],antal_ak.max(),2)
    o1 = omsattning(antal_ak[i],antal_ak.max(),0)
    o2 = omsattning(antal_ak[i],antal_ak.max(),1)
    o3 = omsattning(antal_ak[i],antal_ak.max(),2)
    v1 = vinst(antal_ak[i],antal_ak.max(),0)
    v2 = vinst(antal_ak[i],antal_ak.max(),1)
    v3 = vinst(antal_ak[i],antal_ak.max(),2)
    ak_kostnader.append(k1)
    tk_kostnader.append(k2)
    fk_kostnader.append(k3)
    ak_omsattningar.append(o1)
    tk_omsattningar.append(o2)
    fk_omsattningar.append(o3)
    ak_vinster.append(v1)
    tk_vinster.append(v2)
    fk_vinster.append(v3)

#Plotta 2:a Klass
plt.plot(antal_ak,ak_kostnader,"-r")
plt.plot(antal_ak, ak_omsattningar, "-b")
plt.plot(antal_ak,ak_vinster, "-g")
ind = ak_vinster.index(np.max(ak_vinster))
plt.plot(antal_ak[ind],ak_vinster[ind],"om")
maxi = f"Maximal vinst är: \n {round(ak_vinster[ind],0)} SEK"
plt.text(antal_ak[ind],ak_vinster[ind]-3*10**4,maxi, ha ='center')
prisi = f"Priset på en biljett för 2a:Klass är: \n {round(pris(antal_ak[ind],np.max(antal_ak),0),0)} SEK"
plt.text(antal_ak[ind],ak_vinster[ind]-5.5*10**4, prisi, ha ='center')
plt.legend(["Kostnad","Intäkt","Vinst"])
plt.xlabel('Antal Resenärer')
plt.ylabel('Kronor(SEK)')
plt.show()

#Plotta Tyst Kupé
plt.plot(antal_ak, tk_kostnader,"-r")
plt.plot(antal_ak, tk_omsattningar, "-b")
plt.plot(antal_ak, tk_vinster, "-g")
ind = tk_vinster.index(np.max(tk_vinster))
plt.plot(antal_tk[ind],tk_vinster[ind],"om")
maxi = f"Maximal vinst är: \n {round(tk_vinster[ind],0)} SEK"
plt.text(antal_tk[ind],tk_vinster[ind]-0.5*10**5,maxi, ha ='center')
prisi = f"Priset på en biljett för Tyst Kupé är: \n {round(pris(antal_tk[ind],np.max(antal_tk),1),0)} SEK"
plt.text(antal_tk[ind],tk_vinster[ind]-1*10**5,prisi, ha ='center')
plt.legend(["Kostnad","Intäkt","Vinst"])
plt.xlabel('Antal Resenärer')
plt.ylabel('Kronor(SEK)')
plt.show()

#Plotta 1a:Klass
plt.plot(antal_ak, fk_kostnader,"-r")
plt.plot(antal_ak, fk_omsattningar, "-b")
plt.plot(antal_ak, fk_vinster, "-g")
ind = fk_vinster.index(np.max(fk_vinster))
plt.plot(antal_fk[ind],fk_vinster[ind],"om")
maxi = f"Maximal vinst är: \n {round(fk_vinster[ind],0)} SEK"
plt.text(antal_fk[ind],fk_vinster[ind]-1.5*10**5,maxi, ha ='center')
prisi = f"Priset på en biljett för 1a:Klass är: \n {round(pris(antal_fk[ind],np.max(antal_fk),1),0)} SEK"
plt.text(antal_fk[ind],fk_vinster[ind]-3*10**5, prisi, ha ='center')
plt.legend(["Kostnad","Intäkt","Vinst"])
plt.xlabel('Antal Resenärer')
plt.ylabel('Kronor(SEK)')
plt.show()

##Vi definierar våran funktion för total vinst
def total_vinst(platser, plats):
    v1 = vinst(platser[0],plats,0)
    v2 = vinst(platser[1],plats,1)
    v3 = vinst(platser[2],plats,2)
    v = v1 + v2 + v3
    return v

#Skapr listor för våran optimering
x_bas = [0.3,0.4,0.3]
platser = np.linspace(100,1000,10)
opt_platser = []
vinster = []
opt_fordelning = []

#Nu tar vi fram våra maximala vinster per antal platser
for plats in platser:
    x0 = [x_bas[0]*plats,x_bas[1]*plats,x_bas[2]*plats]
    def min_vinst(x):
        max = total_vinst(x, plats)
        return max*-1
    bounds =  [(0, plats),(0,plats),(0,plats)]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - plats})
    opt_pris = optimize.minimize(fun = min_vinst, x0 = x0,method='SLSQP', options ={'ftol': 1e-9}, bounds=bounds, constraints=constraints)
    platsen = opt_pris.x
    for i in range(len(platsen)):
        platsen[i] = round(platsen[i],2)
    vinsten = round(opt_pris.fun*-1,0)
    opt_platser.append(platsen)
    opt_fordelning.append(platsen/plats)
    vinster.append(vinsten)

#Plotta våra maximala vinster
plt.plot(platser, vinster, "bo")
plt.legend(["Vinst"])
plt.xlabel('Antal Resenärer')
plt.ylabel('Kronor(SEK)')
plt.show()

#Skapar en tabell med våra resultat
tab = pd.DataFrame(data = opt_platser, index = platser,columns = ["2a:Klass", "Tyst Kupé","1:aKlass"])
tab['Max Vinst'] = vinster

#Spara tabellen tlll latex
with open('Optimeringsresultat.tex', 'w') as tf:
    tf.write(tab.to_latex())

#Nu testa vi lite mer begränsningar
platsers = np.linspace(10,500,50)
opt_platsers = []
vinsters = []
opt_fordelnings = []

for i in range(1,51):
    plats = 1000
    x0 = [x_bas[0]*i*plats/50,x_bas[1]*i*plats/50,x_bas[2]*i*plats/50]
    def min_vinst(x):
        max = total_vinst(x, plats)
        return max*-1
    bounds =  [(0, plats),(0,plats),(0,plats)]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - i*plats/50})
    opt_pris = optimize.minimize(fun = min_vinst, x0 = x0,method='SLSQP', options ={'ftol': 1e-9}, bounds=bounds, constraints=constraints)
    platsen = opt_pris.x
    vinsten = opt_pris.fun*-1
    opt_platsers.append(platsen)
    opt_fordelnings.append(platsen/plats)
    vinsters.append(vinsten)

for i in range(len(platsers)):
    platsers[i] = platsers[i]/platsers.max()
    vinsters[i] = vinsters[i]/np.max(vinsters)

plt.plot(platsers, vinsters, "bo")
plt.legend(["Vinst"])
plt.xlabel('Antal Resenärer')
plt.ylabel('Kronor(SEK)')
plt.show()

#Plottar våra kombinationer av antal passagerare
def z(x,y):
    z = 500 - x - y 
    return z

xvarden = np.arange(0,501)
yvarden = np.arange(0,501)

X, Y = np.meshgrid(xvarden,yvarden)
Z = z(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X,Y,Z,100,cmap='binary')
ax.set_xlabel('Antal Resenärer: 2a:Klass')
ax.set_ylabel('Antal Resenärer: Tyst Kupé')
ax.set_zlabel('Antal Resenärer: 1a:Klass')

ax.view_init(0, -35)
fig
Z = pd.DataFrame(Z)

for i in range(len(Z)):
    for j in range(len(Z[0])):
        if Z.iloc[i,j] <= -1:
            Z.iloc[i,j] = np.nan

fig = plt.figure(figsize=(10, 14))
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, z(X,Y)*0)
ax.plot_surface(X, Y, Z)
ax.set_xlabel('Antal Resenärer: 2a:Klass')
ax.set_ylabel('Antal Resenärer: Tyst Kupé')
ax.set_zlabel('Antal Resenärer: 1a:Klass')
ax.set_zlim(-100, 500)
ax.view_init(25, 25)
plt.show()

xpriser = pris(xvarden,xvarden.max(),0)

ypriser = pris(yvarden,yvarden.max(),1)

X1, Y1 = np.meshgrid(xpriser,ypriser)
Z = z(X,Y)
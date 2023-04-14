import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as polynomial
import scipy.optimize as optimize
import math 
import pandas as pd
import tabulate as tab
from IPython.display import display
from pandas.plotting import table
import googlemaps
 
#Google maps API
gmaps = googlemaps.Client(key='AIzaSyDUSFRYXXmwU7MPuwQw4C1OqCdY0IujtaU')

#Total konsumtionen för vår konsumentbas
tot_kons = 1000

#Producenter: x-koordinater, y-koordinater, produktionsandel, namn på städer
malmo = [13.00,55.60,0.2,"Malmö"]
stockholm = [18.07,59.33,0.35,"Stockholm"]
goteborg = [11.97,57.71,0.45,"Göteborg"]
producenter = [stockholm,goteborg,malmo]

#Grossister: x-koordinater, y-koordinater, konsumtionsandel, namn på städer
umea = [20.23,63.83,132235,"Umeå"]
linkoping = [15.62,58.41,166673,"Linköping"]
orebro = [15.21,59.28,158057,"Örebro"]
karlstad = [13.51,59.40,96466,"Karlstad"]
kiruna = [20.26,67.86,22243,"Kiruna"]
kalmar = [16.36,56.66,72018,"Kalmar"]
ostersund = [14.64,63.18,64714,"Östersund"]
helsingborg = [12.69,56.05,150975,"Helsingborg"]
grossister = [helsingborg, kalmar, kiruna, umea, ostersund, karlstad, orebro, linkoping]

#Listor för producenter, x-koordinater, y-koordinater, namn på städer, producentnivåer
x_pro = []
y_pro = []
label_pro = []
produktniva = []

#Listor för grossister, x-koordinater, y-koordinater, namn på städer, konsumtionsnivåer
x_gro = []
y_gro = []
label_gro = []
konsumt_niva = []

#Lägga till namn i grafen
def addlabels(x,y,s):
    for i in range(len(x)):
        plt.text(x[i], y[i]+0.15, s[i], ha = 'center')

#Skapa listor för producenter
for i in range(len(producenter)):
    x = producenter[i][0]
    y = producenter[i][1]
    niva = producenter[i][2]*tot_kons
    label = producenter[i][3]
    x_pro.append(x)
    y_pro.append(y)
    label_pro.append(label)
    produktniva.append(niva)

#Skapa listor för grossister
for i in range(len(grossister)):
    x = grossister[i][0]
    y = grossister[i][1]
    pop = grossister[i][2]
    label = grossister[i][3]
    x_gro.append(x)
    y_gro.append(y)
    label_gro.append(label)
    konsumt_niva.append(pop)

#Avrundning av våra konsumptionsvärden
totpop = np.sum(konsumt_niva)
konsumt_niva = konsumt_niva/totpop*tot_kons
for i in range(len(konsumt_niva)):
    konsumt_niva[i]= round(konsumt_niva[i],0)

#Plotta alla ställen på en karta
plt.figure(figsize=(8,12))
plt.plot(x_pro,y_pro,'ro')
addlabels(x_pro,y_pro,label_pro)
plt.plot(x_gro,y_gro,'bo')
addlabels(x_gro,y_gro,label_gro)
plt.legend(['Producenter','Grossister'])
plt.xlabel('Longitud (Ö)')
plt.ylabel('Latitud (N)')

#Fixa till grafen med axlar
#f = plt.figure(figsize=(8,16))
#f.set_figwidth(8)
#f.set_figheight(16)
plt.show()

#Skapa matris och array för våra direktioner och avstånd för alla grossister till respektive producenter
direktioner = []
avstands_matris = np.ones((len(producenter),len(grossister)))

#Avståndsmatris
for i in range(len(grossister)):
    for j in range(len(producenter)):
        g = label_gro[i]
        p = label_pro[j]
        dist = gmaps.distance_matrix(g,p)['rows'][0]['elements'][0]['distance']['value']
        avstands_matris[j][i] = dist

#Skapa en dataframe med alla avstånd
avstandet = pd.DataFrame(avstands_matris, index=label_pro, columns=label_gro)

#Utsläpp per 
konsumptions_matris = np.ones((len(producenter),len(grossister)))
konsumptions_matris[0] = konsumt_niva
konsumptions_matris[1] = konsumt_niva
konsumptions_matris[2] = konsumt_niva
utslapps_matris = konsumptions_matris*avstands_matris

#Vårt totala avstånd givet en matris
def total_utslapp(matris):
    matris_avstand = utslapps_matris*matris
    x = np.sum(matris_avstand)
    print(x)
    return x

#Hur vi tar fram minimalt avståend givet en array
def min_utslapp(array):
    matris = skapa_matris(array,3,8)
    avs = total_utslapp(matris)
    return avs

#Funktion för att returnera en matris med givna antal rader och kolumner från en array 
def skapa_matris(array,r,c):
    matris = np.zeros((r,c))
    for i in range(r):
        l = len(array)/r
        matris[i]=array[int(i*l):int(l*(i+1))]
    return matris

#Vår första gissning på direktioner i form av en array
arr = [0,0,1,1,1,0,96/183,0,0,58/83,0,0,0,1,87/183,1,1,25/83,0,0,0,0,0,0]
xbas = arr

#Vilka värden våra x-värden kan ta
##Direktioner kan endast vara positiva och maximalt 100%
bounds_alla = [(0, 1)]*24

#Våra begränsningar till vårat optimeringsproblem
##Radsummorna ska vara lika med respektive produktionsnivåer
###Alla kolumnsummor ska vara lika med 1
cons_alla = ({'type': 'eq', 'fun': lambda x: np.matmul(x[0:8],konsumt_niva) - produktniva[0]},
        {'type': 'eq', 'fun': lambda x: np.matmul(x[8:16],konsumt_niva) - produktniva[1]},
        {'type': 'eq', 'fun': lambda x: np.matmul(x[16:],konsumt_niva) - produktniva[2]},
        {'type': 'eq', 'fun': lambda x: x[0] + x[8] + x[16] - 1},
        {'type': 'eq', 'fun': lambda x: x[1] + x[9] + x[17] - 1},
        {'type': 'eq', 'fun': lambda x: x[2] + x[10] + x[18] - 1},
        {'type': 'eq', 'fun': lambda x: x[3] + x[11] + x[19] - 1},
        {'type': 'eq', 'fun': lambda x: x[4] + x[12] + x[20] - 1},
        {'type': 'eq', 'fun': lambda x: x[5] + x[13] + x[21] - 1},
        {'type': 'eq', 'fun': lambda x: x[6] + x[14] + x[22] - 1},
        {'type': 'eq', 'fun': lambda x: x[7] + x[15] + x[23] - 1})

#Optimeringsproblemet
fördelning = optimize.minimize(fun=min_utslapp, x0=xbas, method='SLSQP', options ={'ftol': 1e-9}, bounds=bounds_alla, constraints=cons_alla)

#Loopar för att minimera 
for i in range(10):
    xbas = fördelning.x
    y = optimize.minimize(fun=min_utslapp, x0=xbas, method='SLSQP', options ={'ftol': 1e-9}, bounds=bounds_alla, constraints=cons_alla)
    if min_utslapp(fördelning.x)>min_utslapp(y.x):
        fördelning = y
    else:
        break

#Totalsträcka
optimal_sträcka = min_utslapp(fördelning.x)

#Direktioner för att sända ut varor
optimal_matris = skapa_matris(fördelning.x,3,8)

#Funktion för en levereringsmatris    
def levererings_matris(matris):
    kons = np.zeros((len(matris),len(matris[0])))  
    for o in range(len(matris)):
        kons[o] = matris[o]*konsumt_niva
    for i in range(len(matris)):
        for j in range(len(matris[0])):
            kons[i][j] = round(kons[i][j],0)
    return kons
optimala_leveranser = levererings_matris(optimal_matris)

#Skapa en tabell av våra resultat
optimal_tabell = pd.DataFrame(optimal_matris, index=label_pro, columns=label_gro)
levererings_tabell = pd.DataFrame(optimala_leveranser, index=label_pro, columns=label_gro)

#Städers färger
colours = ['b--','r--','y--']

#Rita linjer
plt.figure(figsize=(8,12))
plt.plot(x_pro,y_pro,'ro')
addlabels(x_pro,y_pro,label_pro)
plt.plot(x_gro,y_gro,'bo')
addlabels(x_gro,y_gro,label_gro)
plt.legend(['Producenter','Grossister','Lager'])
plt.xlabel('Longitud (Ö)')
plt.ylabel('Latitud (N)')
for i in range(len(optimala_leveranser)):
    for j in range(len(optimala_leveranser[i])):
        if optimala_leveranser[i][j] != 0:
            plt.plot([producenter[i][0],grossister[j][0]],[producenter[i][1],grossister[j][1]],colours[i])
            #plt.text(np.abs(producenter[i][0]+grossister[j][0])/2,np.abs(producenter[i][1]+grossister[j][1])/2,kons[i][j])
longituder = []
latituder = []
värden = []
for i in range(len(optimala_leveranser)):
    for j in range(len(optimala_leveranser[i])):
        if optimala_leveranser[i][j] != 0:
            long = (producenter[i][0]+grossister[j][0])/2
            lat =  (producenter[i][1]+grossister[j][1])/2
            värde = optimala_leveranser[i][j] 
            longituder.append(long)
            latituder.append(lat)
            värden.append(int(värde))
addlabels(longituder,latituder,värden)
plt.show()

#Plotta tabellerna 
optimal_tabell_formaterad = optimal_tabell
for i in range(len(levererings_tabell)):
    for j in range(len(levererings_tabell.iloc[i])):
        x = round(levererings_tabell.iloc[i,j]/konsumt_niva[j]*100,2)
        optimal_tabell_formaterad.iloc[i,j] = '{:.2f}%'.format(x)
plt.figure(figsize=(8,0.25))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)  
table(ax, optimal_tabell_formaterad) 
plt.show()
plt.savefig('direktioner.png')

#Leveranserna
plt.figure(figsize=(8,0.25))
ax = plt.subplot(111, frame_on=False) 
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)  
table(ax, levererings_tabell)
plt.show()
plt.savefig('leveranser')
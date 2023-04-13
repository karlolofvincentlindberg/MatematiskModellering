import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as polynomial
import scipy.optimize as optimize
import math 
import pandas as pd
import tabulate as tab
from IPython.display import display
from pandas.plotting import table

#Hur många kilometer är en breddgrad
bg = 111 

#Total konsumtionen för vår konsumentbas
tot_kons = 1000

#Producenter: x-koordinater, y-koordinater, produktionsandel, namn på städer
malmo = [13.00,55.60,0.2,"Malmö"]
stockholm = [18.07,59.33,0.35,"Stockholm"]
goteborg = [11.97,57.71,0.45,"Göteborg"]
producenter = [goteborg,malmo,stockholm]

#Grossister: x-koordinater, y-koordinater, konsumtionsandel, namn på städer
umea = [20.23,63.83,132235,"Umeå"]
linkoping = [15.62,58.41,166673,"Linköping"]
orebro = [15.21,59.28,158057,"Örebro"]
karlstad = [13.51,59.40,96466,"Karlstad"]
kiruna = [20.26,67.86,22243,"Kiruna"]
kalmar = [16.36,56.66,72018,"Kalmar"]
ostersund = [14.64,63.18,64714,"Östersund"]
helsingborg = [12.69,56.05,150975,"Helsingborg"]
grossister = [helsingborg, kalmar, karlstad, kiruna, linkoping, umea, orebro, ostersund]

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

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Skillnader i lat o long
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # haversine av halv distans
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    
    # distans
    r = 6371  # jorden
    d = 2 * r * np.arcsin(np.sqrt(a))
    
    return d

#Avståndsmatris
for i in range(len(grossister)):
        x = x_gro[i]
        y = y_gro[i]
        min = 1000
        ind = 0
        for j in range(len(producenter)):
           #Ändra ekvation för avstånd 
            avsta = haversine(y_gro[i],x_gro[i],y_pro[j],x_pro[j])
            avstands_matris[j][i] = avsta

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



##Nu Introducerar vi ett möjligt lager
#Formel för det totala utsläppet
def total_utslapp_lager(matris, utslappsniva):
    tot = matris*utslappsniva
    return np.sum(tot)

#Formel för att beräkna ut det totala utsläppet per resa
def utslappsnivaer_lager(lager, gross, prod, utslb, utsta):
    lag_lat = lager[0]
    lag_lon = lager[1]
    matris = np.zeros((len(prod)*2,len(gross)))
    mat = np.zeros((len(prod)*2,len(gross)))
    for i in range(len(prod)):
        for j in range(len(gross)):
            p = prod[i]
            g = gross[j]
            matris[i][j] = haversine(p[0], p[1], g[0], g[1],)*utslb
    for i in range(len(prod)):
        for j in range(len(gross)):
            p = prod[i]
            g = gross[j]
            uts1 = haversine(p[0], p[1], lag_lat, lag_lon)*utsta
            uts2 = haversine(lag_lat, lag_lon, g[0], g[1],)*utslb
            matris[i+len(prod)][j] = uts1 + uts2
    for i in range(len(matris)):
        for j in range(8):
            mat[i][j] = matris[i][j]*konsumt_niva[j]
    return mat 

#Första gissning på x-värden
arr_lager = arr
for i in range(len(arr_lager)):
    arr_lager.append(0)

#Funktion för att finna det mest optimala lagret 
def optimal_lager(lager, gross, prod, utslb, utsta, x0, bounds, cons):
    utslapp = utslappsnivaer_lager(lager,gross, prod, utslb, utsta)
    def func(array):
        matris = np.zeros((len(prod)*2,len(gross)))
        for i in range(len(prod)*2):
            l = len(gross)
            matris[i] = array[l*i:(i+1)*l]
        tot = total_utslapp_lager(matris, utslapp)
        return tot
    res = optimize.minimize(fun=func, x0=x0, method='SLSQP', options ={'ftol': 1e-9}, bounds=bounds, constraints=cons)
    for i in range(10):
        x0 = res.x 
        y = optimize.minimize(fun=func, x0=x0, method='SLSQP', options ={'ftol': 1e-9}, bounds=bounds, constraints=cons)
        if func(res.x)>func(y.x):
            res = y
        else:
            break
    print(utslapp)
    opt_matris = skapa_matris(res.x, len(producenter)*2,len(grossister))
    result = func(res.x)                          
    return  opt_matris, result

#Begränsningarna
cons_lager = ({'type': 'eq', 'fun': lambda x: np.matmul(x[0:8],konsumt_niva)+np.matmul(x[24:32],konsumt_niva) - produktniva[0]},
        {'type': 'eq', 'fun': lambda x: np.matmul(x[8:16],konsumt_niva) +np.matmul(x[32:40],konsumt_niva)- produktniva[1]},
        {'type': 'eq', 'fun': lambda x: np.matmul(x[16:24],konsumt_niva)+np.matmul(x[40:48],konsumt_niva) - produktniva[2]},
        {'type': 'eq', 'fun': lambda x: x[0] + x[8] + x[16] + x[24] + x[32] + x[40] - 1},
        {'type': 'eq', 'fun': lambda x: x[1] + x[9] + x[17] + x[25] + x[33] + x[41]- 1},
        {'type': 'eq', 'fun': lambda x: x[2] + x[10] + x[18] + x[26] + x[34] + x[42]- 1},
        {'type': 'eq', 'fun': lambda x: x[3] + x[11] + x[19] + x[27] + x[35] + x[43]- 1},
        {'type': 'eq', 'fun': lambda x: x[4] + x[12] + x[20] + x[28] + x[36] + x[44]- 1},
        {'type': 'eq', 'fun': lambda x: x[5] + x[13] + x[21] + x[29] + x[37] + x[45]- 1},
        {'type': 'eq', 'fun': lambda x: x[6] + x[14] + x[22] + x[30] + x[38] + x[46]- 1},
        {'type': 'eq', 'fun': lambda x: x[7] + x[15] + x[23] + x[31] + x[39] + x[47]- 1})

#Begränsningar för x-värden
bounds_lager = [(0, 1)]*48

#Bestämmande av våra lager 
falun = [15.37,60.36,"Falun"]
almhult = [14.08,56.33,"Almhult"]
sundsvall = [17.19,62.24,"Sundsvall"]
jokkmokk = [19.50,66.37,"Jokkmokk"]
jonkoping = [14.09,57.46,"Jonkoping"]

lagren = [falun, almhult, sundsvall, jokkmokk, jonkoping]

#Våra lagerlistor
lager_matriser = []
lager_utslapp = []
lev_lager_matriser = []
lev_lager_namn = []

#Utsläpp per transportmedel
lastbil = 1
tåg = 0.5

#Ta fram rader och kolumner för våran data
def prep_df(lager):
    col = label_gro
    row = []
    for i in range(len(label_pro)):
        st = label_pro[i]
        row.append(st)
    for i in range(len(label_pro)):
        st = label_pro[i]+" via " + lager[2]
        row.append(st)
    return col, row

#Funktion för att ta fram våra optimeringsresultat
for i in range(len(lagren)):
    lager_matris, utslappet = optimal_lager(lagren[i], grossister, producenter, lastbil, tåg, arr_lager, bounds_lager, cons_lager)
    levering_lager = levererings_matris(lager_matris)
    lager_matriser.append(lager_matris)
    lager_utslapp.append(utslappet)
    lev_lager_matriser.append(levering_lager)
    lev_lager_namn.append((lagren[i])[2])


#Funktion för att transformera våra resultat till data
def trans_df(i):
    lager_matris = lager_matriser[i]
    lev_matris = lev_lager_matriser[i]
    lager = lagren[i]
    col, row = prep_df(lager)
    for i in range(len(row)):
        for j in range(len(col)):
            lager_matris[i][j] = round(lager_matris[i][j],2)
    for i in range(len(row)):
        for j in range(len(col)):
            lev_matris[i][j] = round(lev_matris[i][j],2)    
    df1 = pd.DataFrame(data=lager_matris,columns=col,index=row)
    df2 = pd.DataFrame(data=lev_matris,columns=col,index=row)
    return df1, df2

dataframes = []

#skapa en massa data
for j in range(4):
    d1, d2 = trans_df(j)
    dataframes.append(d1)
    dataframes.append(d2)

colours = ['b--','r--','y--']

#Plotta valet
def plot_lager(i):
    lager = lagren[i]
    matris = lev_lager_matriser[i]
    plt.figure(figsize=(8,12))  
    plt.plot(x_pro,y_pro,'ro')
    addlabels(x_pro,y_pro,label_pro)
    plt.plot(x_gro,y_gro,'bo')
    addlabels(x_gro,y_gro,label_gro)
    plt.plot(lager[0],lager[1],'go')
    plt.text(lager[0],lager[1]+0.15,lager[2], ha ='center')
    plt.legend(['Producenter','Grossister','Lager'])
    plt.xlabel('Longitud (Ö)')
    plt.ylabel('Latitud (N)')
    plt.text(16,67,"Totalt utsläpp:",ha = 'center')
    plt.text(16,66.7,str(round(lager_utslapp[i],2))+" ",ha = 'center')
    for i in range(len(matris)):
        for j in range(len(matris[i])):
            if matris[i][j] != 0 and i<3:
                plt.plot([producenter[i][0],grossister[j][0]],[producenter[i][1],grossister[j][1]],colours[i])
            if matris[i][j] != 0 and i>=3:
                plt.plot([producenter[i-3][0],lager[0]],[producenter[i-3][1],lager[1]],colours[i-3])
                plt.plot([lager[0],grossister[j][0]],[lager[1],grossister[j][1]],"g--")
    longituder = []
    latituder = []
    värden = []
    for i in range(int(len(matris)/2)):
        for j in range(len(matris[i])):
            if matris[i][j] != 0 and i<3:
                long = (producenter[i][0]+grossister[j][0])/2
                lat =  (producenter[i][1]+grossister[j][1])/2
                värde = matris[i][j] 
                longituder.append(long)
                latituder.append(lat)
                värden.append(int(värde))
    for i in range(3,len(matris)):
        if np.sum(matris[i,:]) != 0:
            long1 = (producenter[i-3][0]+lager[0])/2
            lat1 =  (producenter[i-3][1]+lager[1])/2
            värde1 = np.sum(matris[i,:]) 
            longituder.append(long1)
            latituder.append(lat1)
            värden.append(int(värde1))
        for j in range(len(matris[i])):
            if np.sum(matris[3:,j]) != 0 and i<4:
                long2 = (lager[0]+grossister[j][0])/2
                lat2 =  (lager[1]+grossister[j][1])/2
                värde2 = np.sum(matris[3:,j])
                longituder.append(long2)
                latituder.append(lat2)
                värden.append(int(värde2))
    addlabels(longituder,latituder,värden)
    

for i in range(len(lagren)):
    plot_lager(i)
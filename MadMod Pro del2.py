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
from datetime import date
from pandas import ExcelWriter
 
def save_xls(list_dfs, labels, xls_path):
    with ExcelWriter(xls_path)as writer:
        for i in range(len(list_dfs)):
            list_dfs[i].to_excel(writer, sheet_name=labels[i])

# Requires API key
gmaps = googlemaps.Client(key='AIzaSyDUSFRYXXmwU7MPuwQw4C1OqCdY0IujtaU')
 
# Requires cities name
def distance(p1, p2):
    my_dist = gmaps.distance_matrix(p1,p2)['rows'][0]['elements'][0]['distance']['value']
    return my_dist

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

#Funktion för att returnera en matris med givna antal rader och kolumner från en array 
def skapa_matris(array,r,c):
    matris = np.zeros((r,c))
    for i in range(r):
        l = len(array)/r
        matris[i]=array[int(i*l):int(l*(i+1))]
    return matris

#Funktion för en levereringsmatris    
def levererings_matris(matris):
    kons = np.zeros((len(matris),len(matris[0])))  
    for o in range(len(matris)):
        kons[o] = matris[o]*konsumt_niva
    for i in range(len(matris)):
        for j in range(len(matris[0])):
            kons[i][j] = round(kons[i][j],0)
    return kons

#Städers färger
colours = ['b--','r--','y--']

##Nu Introducerar vi ett möjligt lager
#Formel för det totala utsläppet
def total_utslapp_lager(matris, utslappsniva):
    tot = matris*utslappsniva
    return np.sum(tot)

avstans = []

#Formel för att beräkna ut det totala utsläppet per resa
def utslappsnivaer_lager(lager, gross, prod, utslb, utsta):
    lager_namn = lager[2]
    l = len(prod)
    mat = np.zeros((len(prod)*3,len(gross)))
    matris = np.zeros((len(prod)*2,len(gross)))
    for i in range(len(prod)):
        for j in range(len(gross)):
            dist = distance(label_pro[i], label_gro[j])
            mat[i][j] = dist
            matris[i][j] = dist*utslb*konsumt_niva[j]
    for i in range(len(prod)):
        for j in range(len(gross)):
            dist1 = distance(label_pro[i], lager_namn)
            dist2 = distance(lager_namn, label_gro[j])
            mat[i+l][j] = dist1
            mat[i+2*l][j] = dist2
            matris[i+l][j] = (dist1*utsta+dist2*utslb)*konsumt_niva[j]
    avstans.append(mat)
    return matris 

utslappen = []
for i in range(len(lagren)):
    x = utslappsnivaer_lager(lagren[i], grossister, producenter, lastbil, tåg)
    utslappen.append(x)

#Första gissning på x-värden
arr_1 = [0,0,1,0,0,0,96/183,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,58/83,0,0,0,0,87/183,0,0,25/83,0,0,0,0,0,0]
arr_2 = [0,0,1,1,1,0,96/183,0,0,58/83,0,0,0,1,87/183,1,1,25/83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
arr_3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,96/183,0,0,58/83,0,0,0,1,87/183,1,1,25/83,0,0,0,0,0,0]

arr_4 = [0,0,1,1,0,0,96/183,0,0,58/83,0,0,0,1,87/193,1,1,25/83,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
arr_5 = [0,0,1,1,1,0,96/183,0,0,0,0,0,0,1,87/183,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,58/83,0,0,0,0,0,0,0,25/83,0,0,0,0,0,0]
arr_6 = [0,0,0,0,0,0,96/183,0,0,58/83,0,0,0,1,87/183,1,1,25/83,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
arr_7 = [0,0,0,1,1,0,96/183,0,0,58/83,0,0,0,1,87/193,1,1,25/83,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
arr_8 = [0,0,1,1,1,0,96/183,0,0,0,0,0,0,1,87/193,0,1,25/83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,56/83,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
arrs = [arr_1, arr_2, arr_3, arr_4, arr_5, arr_6, arr_7, arr_8]

#Funktion för att finna det mest optimala lagret 
def optimal_lager(utslapp, gross, prod, x0, bounds, cons):
    def func(array):
        matris = np.zeros((len(prod)*2,len(gross)))
        for i in range(len(prod)*2):
            l = len(gross)
            matris[i] = array[l*i:(i+1)*l]
        tot = total_utslapp_lager(matris, utslapp)
        return tot
    res = optimize.minimize(fun=func, x0=x0, method='SLSQP', options ={'ftol': 1e-9}, bounds=bounds, constraints=cons)
    for i in range(len(arrs)):
        x0 = arrs[i]
        y = optimize.minimize(fun=func, x0=x0, method='SLSQP', options ={'ftol': 1e-9}, bounds=bounds, constraints=cons)
        if func(res.x)>func(y.x):
            res = y
    opt_matris = skapa_matris(res.x, len(producenter)*2,len(grossister))
    result = func(res.x) 
    print(res.x) 
    print("Stopp")                        
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

#Funktion för att ta fram våra optimeringsresultat
for i in range(len(lagren)):
    lager_matris, utslappet = optimal_lager(utslappen[i], grossister, producenter, arrs[0], bounds_lager, cons_lager)
    levering_lager = levererings_matris(lager_matris)
    lager_matriser.append(lager_matris)
    lager_utslapp.append(utslappet)
    lev_lager_matriser.append(levering_lager)
    lev_lager_namn.append((lagren[i])[2])

#Ändringar
#Falun
lev_lager_matriser[0][1][6] = 87

#Jokkmokk
lev_lager_matriser[3][1][6] = 87

#Jönköping
lev_lager_matriser[4][4][1] = 58
lev_lager_matriser[4][1][6] = 87


#Funktion för att transformera våra resultat till data
def trans_df(i):
    lager_matris = lager_matriser[i]
    lev_matris = lev_lager_matriser[i]
    utslapp = utslappen[i]
    avsta = avstans[i]
    lager = lagren[i]
    col, row = prep_df(lager)
    
    for i in range(len(row)):
        for j in range(len(col)):
            lager_matris[i][j] = round(lager_matris[i][j],2)
    for i in range(len(row)):
        for j in range(len(col)):
            lev_matris[i][j] = round(lev_matris[i][j],2) 
        for i in range(len(row)):
            for j in range(len(col)):
                utslapp[i][j] = round(utslapp[i][j],2)      
    df1 = pd.DataFrame(data=lager_matris,columns=col,index=row)
    df2 = pd.DataFrame(data=lev_matris,columns=col,index=row)
    df3 = pd.DataFrame(data=utslapp,columns=col,index=row)
    df4 = pd.DataFrame(data=avsta,columns=col,index=row+[7,8,9])
    return df1, df2, df3, df4

dataframes = []
datalabels = []

#skapa en massa data
for j in range(5):
    d1, d2, d3, d4 = trans_df(j)
    dataframes.append(d1)
    dataframes.append(d2)
    dataframes.append(d3)
    dataframes.append(d4)
    datalabels.append("lager"+lagren[j][2])
    datalabels.append("lever"+lagren[j][2])
    datalabels.append("utslapp"+lagren[j][2])
    datalabels.append("avstand"+lagren[j][2])

#save_xls(dataframes,datalabels, "wow.xlsx")

colours = ['m--','c--','y--']

#Plotta valet
def plot_lager(i):
    lager = lagren[i]
    matris = lev_lager_matriser[i]
    plt.figure(figsize=(8,12))  
    plt.plot(x_pro,y_pro,'ro')
    addlabels(x_pro,y_pro,label_pro)
    plt.plot(x_gro,y_gro,'bv')
    addlabels(x_gro,y_gro,label_gro)
    plt.plot(lager[0],lager[1],'ks')
    plt.text(lager[0],lager[1]+0.15,lager[2], ha ='center')
    plt.legend(['Producenter','Grossister','Lager'])
    plt.xlabel('Longitud (Ö)')
    plt.ylabel('Latitud (N)')
    #plt.text(16,67,"Totalt utsläpp: ",ha = 'center')
    #plt.text(16,66.7,str(round(lager_utslapp[i]/10**6,2))+" 10^6",ha = 'center')
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

# ---------------------------------------------------------------------------------------------------------------------
# Modified copy of InstellCa_3.0.py (InstellCa, Exoplanet Instellation Calculator)
#   Original: https://github.com/Mradumay137/InstellCa, commit 99225f4 (file last changed 2025-09-15),
#   Copyright (c) 2021 Mradumay137 (Mradumay Sadh), MIT License. Reference: Sadh & Gavassino 2026, ApJ 997, 167.
#   The MIT permission notice is reproduced at the end of this header; the original LICENSE file is at https://github.com/Mradumay137/InstellCa (commit 99225f4).
#
# Modifications (Jonas Spiller, 2026-11-01), marked "# MOD" below; everything else is verbatim:
#   1. The limb-darkening coefficient u (hard-wired to 0.6 in the original) is read from the environment variable
#      INSTELLCA_U (default 0.6), so the u = 0 branch of the original code can be run without editing the file.
#   2. The computed arrays are written to a text file in a subfolder next to this script (they were only plotted in the original):
#      (in the subfolder outfiles/); columns = angle from the substellar point [deg], geometric (extended-source) model [kW/m^2], inverse-square
#      model [kW/m^2]. The figure is saved next to it as well, under the same name.
#   3. Numerical knobs for convergence tests, all defaulting to the original behaviour: INSTELLCA_EPSREL / INSTELLCA_EPSABS /
#      INSTELLCA_LIMIT are passed to both scipy.integrate.quad calls (original: scipy defaults), INSTELLCA_NLAT sets the number
#      of latitude points (original: 500), INSTELLCA_TAG is appended to the output file names.
#   The physics, integration limits, integrand and normalisation are unchanged.
#   The script must still be run with the InstellCa clone as working directory (it reads ./Exoplanet_Catalogs).
#
# MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice
# and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS
# PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ---------------------------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:53:50 2025

@author: maverick
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:00:38 2020

@author: Mradumay
"""
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
import pandas as pd
from colorama import Fore, Style
import os
import glob
import sys
from scipy import integrate
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outfiles")   # MOD: where the exported arrays and the figure go
os.makedirs(OUT_DIR, exist_ok=True)   # MOD
QUAD_KW = {k: float(os.environ[v]) for k, v in (("epsrel", "INSTELLCA_EPSREL"), ("epsabs", "INSTELLCA_EPSABS")) if v in os.environ}   # MOD
if "INSTELLCA_LIMIT" in os.environ: QUAD_KW["limit"] = int(os.environ["INSTELLCA_LIMIT"])   # MOD
NLAT = int(os.environ.get("INSTELLCA_NLAT", "500"))   # MOD: original: 500 latitude points

num=int(input("Enter number of planets to be plotted:"))
num1=round(math.sqrt(num))
col=num1
row=math.ceil(num/num1)
cwd=os.getcwd()
extension = 'csv'
cwd1=cwd+"/Exoplanet_Catalogs"
cwd2=cwd+"/Limb_darkening_data"
os.chdir(cwd1)
files = glob.glob('*.{}'.format(extension))
l=len(files)
namef="exoplanet.eu_catalog.csv"
exo=[]
if num<2:
    fig=plt.figure(figsize=(row**2/2+row+2, col**2+2*col+11),constrained_layout=False)
if num>2:
    fig=plt.figure(figsize=(col**2+2*col+11,row**2/2+row+11), constrained_layout=False)

newa=[]
for te in range(1,num+1):
    itr=0
    for entry in files:
        if entry==namef and l==1:
            catalog=0
            print(Fore.WHITE +"Exoplanet.eu catalog found")
            print(Style.RESET_ALL)
            break
        else:
            catalog=float(input("Enter 0 for Exoplanet.eu, 1 for NASA arxiv, and 2 for entering manually:"))
            break
    if l==0:
        print(Fore.RED +"No catalog found. Enter parameters manually")
        print(Style.RESET_ALL)
        catalog=2
    while catalog!=0 and catalog!=1 and catalog!=2:
            catalog=float(input(Fore.RED +"Wrong option entered. Please re-enter:"))
            print(Style.RESET_ALL)
    if catalog==0:
        for entry in files:
            itr=itr+1
            if entry==namef:
                break
            if entry!=namef and itr==l: 
                sys.exit(Fore.RED +"Exoplanet.eu catalog not found")
                print(Style.RESET_ALL)
        data=pd.read_csv(os.path.join(cwd1,"exoplanet.eu_catalog.csv"))
        df=pd.DataFrame(data)
        starrad=df["star_radius"]
        planrad=df["radius"]
        temp=df["star_teff"]
        semiax=df["semi_major_axis"]
        name=data["# name"]
        eccentricity=data["eccentricity"]  
        Mass=data["star_mass"] 
        metallicity=data["star_metallicity"]
        exoplanet=input("Enter the name of exoplanet:")
        exo.append(exoplanet)
        opt=float(input("Enter 1 if you wish to change st_rad,2 for pl_rad, 3 for Teff, 4 for sm_axis else enter any #: "))
        g=1
        while g!=0:
            for i in range(len(starrad)):
                if name[i]==exoplanet:
                    g=0
                    break
                elif name[i]!=exoplanet and i==len(starrad)-1:
                    exoplanet=input(Fore.RED +"Exoplanet not found. Please check the name and type again:")
                    print(Style.RESET_ALL)
        for i in range(len(starrad)):
            if name[i]==exoplanet:
                rp1=planrad[i]
                rs1=starrad[i]
                fa1=temp[i]
                al1=semiax[i]
                ecc=eccentricity[i]
                M1=Mass[i]
                met=metallicity[i]
        if opt==1 or opt==12 or opt==13 or opt==14:
            rs1=float(input("Enter stellar radius:"))
        if opt==2 or opt==12 or opt==23 or opt==24:
            rp1=float(input("Enter planet radius:"))
        if opt==3 or opt==13 or opt==23 or opt==34:
            fa1=float(input("Enter effective temperature:"))
        if opt==4 or opt==14 or opt==24 or opt==34:
            al1=float(input("Enter semi-major axis:"))
    if catalog==1:
        filename=input("Enter name of NASA arxiv csv file:")
        it=0
        for entry in files:
            it=it+1
            if entry==filename:
                g1=0
                break
        if it==len(files):
            sys.exit(Fore.RED +"File name incorrect or file missing. Please check file or re-type")
            print(Style.RESET_ALL)
        data=pd.read_csv(os.path.join(cwd1,filename),error_bad_lines=False,skiprows=361,low_memory=False)
        df=pd.DataFrame(data)
        planrad=df["pl_radj"]
        starrad=df["st_rad"]
        temp=df["st_teff"]
        semiax=df["pl_orbsmax"]
        name=data["pl_name"]   
        eccentricity=data["pl_orbeccen"] 
        Mass=data["st_mass"]
        metallicity=data["st_metfe"]
        exoplanet=input("Enter the name of exoplanet:")
        exo.append(exoplanet)
        opt=float(input("Enter 1 if you wish to change st_rad,2 for pl_rad, 3 for Teff, 4 for sm_axis else enter any #: "))
        g2=1
        while g2!=0:
            for i in range(len(starrad)):
                if name[i]==exoplanet:
                    g2=0
                    break
                elif name[i]!=exoplanet and i==len(starrad)-1:
                    exoplanet=input(Fore.RED +"Exoplanet not found. Please check the name and type again:")
                    print(Style.RESET_ALL)
        for i in range(len(starrad)):
            if name[i]==exoplanet:
                    rp1=planrad[i]
                    rs1=starrad[i]
                    fa1=temp[i]
                    al1=semiax[i]
                    ecc=eccentricity[i]
                    M1=Mass[i]
                    met=metallicity[i]
        if opt==1 or opt==12 or opt==13 or opt==14:
            rs1=float(input("Enter stellar radius:"))
        if opt==2 or opt==12 or opt==23 or opt==24:
            rp1=float(input("Enter planet radius:"))
        if opt==3 or opt==13 or opt==23 or opt==34:
            fa1=float(input("Enter effective temperature:"))
        if opt==4 or opt==14 or opt==24 or opt==34:
            al1=float(input("Enter semi-major axis:"))
    para=1
    
    while para!=4 and para!=1 and para!=2:
            para=float(input(Fore.RED +'Wrong option entered. Please re-enter:'))
            print(Style.RESET_ALL)
    if catalog==2:
        print(Style.RESET_ALL)
        rp1=float(input("Enter radius of planet in Jupiter radii:"))
        rs1=float(input("Enter radius of the host star in units of solar radius:"))
        fa1=float(input("Enter effective Temperature of host star in K:"))
        al1=float(input("Enter semi-major axis of the planet from the star in AU:"))
        ecc=float(input("Enter eccentricity:"))
        
        exoplanet=input("Enter name:")
        if para==4:
            M1=float(input("Enter stellar mass(solar units):"))
            met=float(input("Enter metallicity[Fe/H]:"))
    if para==1:
        u=float(os.environ.get("INSTELLCA_U", "0.6"))   # MOD: was u=0.6; INSTELLCA_U=0 selects the original u==0 branch
        met=0
        M1=1
    if para==2:
        u1=float(input("Enter bolometric quadratic coefficient(u1):"))
        u2=float(input("Enter bolometric quadratic coefficient(u2):"))   
        met=0
        M1=1
        
    if np.isnan(rs1)==True or np.isnan(fa1)==True or np.isnan(al1)==True:
        print(Fore.RED +"Crucial parameter missing")
        print(Style.RESET_ALL)
    else:    
        if np.isnan(rp1)==True:
            rp1=input(Fore.RED +"Radius of planet is missing. Please enter value in Rj units:")
            print(Style.RESET_ALL)
            rp1=float(rp1)
        if np.isnan(met)==True:
            met=float(input(Fore.RED +"Metallicity[Fe/H] missing in dataset. Enter manually:"))
            print(Style.RESET_ALL)
        if np.isnan(M1)==True:
            M1=float(input(Fore.RED +"Stellar mass missing in dataset. Enter manually:"))
            print(Style.RESET_ALL)
        number=1
        obli=0
        if np.isnan(ecc)==True:
            ecc=0
        elif ecc!=0 and ecc<0.3:
            print(Fore.WHITE +"Eccentric orbit detected, calculating values at periastron \033[1;30;47m")
            print(Style.RESET_ALL)
        elif ecc>0.3:
            number=4
            print(Fore.WHITE +"Highly eccentric orbit(e>0.3). Calculating annual mean \033[1;30;47m")
            print(Style.RESET_ALL)
        true1=np.linspace(0,270,number)
        if te==num:
            print(Fore.WHITE +'Generating Plot, Please wait.. \033[1;30;47m')
        print(Style.RESET_ALL)
        average=[]
        inverse=[]
        for j in range(0,number):
            # Orbital and physical parameters
            true=true1[j]*np.pi/180 
            ob1=float(obli)
            ob=ob1*np.pi/180
            rp1=float(rp1)
            rs1=float(rs1)
            al1=float(al1)
            fa1=float(fa1)
            ecc=float(ecc)
            M=M1*2*10**(30)
            rs=rs1*6.955*10**8
            rp=rp1*6.4*10**6*11.21
            al2=al1*1.496*10**11
            al=al2*(1-ecc**2)/(1+ecc*abs(math.cos(true)))
            d=al-rs-rp
            ch=math.acos(rp/(d+rp))
            s3=(math.asin(abs(rs-rp)/al))
            s1=(np.pi/2+s3)
            s1=math.floor(s1*180/np.pi)
            s1=s1*np.pi/180
            s=np.pi/2
            symp=math.acos((rs+rp)/al)
            la1=np.linspace(-s1,s1,NLAT)   # MOD: was 500
            la2=np.linspace((-s1+ob)*180/np.pi,(s1-ob)*180/np.pi,NLAT)   # MOD: was 500
            surfgrav=100*6.67*10**(-11)*M/rs**(2)
            logg=math.log10(surfgrav)
            oldfor=[]
            final=[]
            denom=[]
            numer=[]
            approx=[]
            
            #Limb Darkening
            if para==1 and u==0.6: 
                fa=fa1*1.0573   
            if para==1 and u==0:
                fa=fa1
            P=5.67*10**(-8)*fa**(4)*4*np.pi*rs**2
            zalist=[]
            areaval=[]
            areaneg=[]
            inla=[]
            for k in range(len(la1)):
                la=la1[k]
                beta=al+rp*math.cos(np.pi-la)
                y1=math.acos((rs**2-rp**2*(math.sin(la))**2)/(beta*rs - rp*math.sin(la)*(math.sqrt(rp**2*(math.sin(la))**2-rs**2+beta**2))))*180/np.pi
                y4=math.acos((rs**2-rp**2*(math.sin(la))**2)/(beta*rs + rp*math.sin(la)*(math.sqrt(rp**2*(math.sin(la))**2-rs**2+beta**2))))*180/np.pi
                y5=math.acos(rs/math.sqrt(al**2+rp**2-2*al*rp*math.cos(la)))*180/np.pi
                y6=math.acos((rs+rp*abs(math.sin(la)))/al)
                y=(y1)*np.pi/180
                y2=(y4)*np.pi/180
                y3=(y5)*np.pi/180
                y7=math.acos(rs/al)
                y=y7
                ad1=180*math.atan(rs*math.sin(y)/(d+rs-(rs*math.cos(y))))/np.pi
                ad=math.floor(ad1)*np.pi/180
                vis=math.acos(rs/(math.sqrt(al**2+rp**2-2*al*rp*math.cos(la))))
                P1=5.67*10**(-8)*fa1**(4)*4*np.pi*(rs)**2
            
                y2=y6
                
                #General integral function
                def function(x,th,la): 
                    a=-rs*math.sin(th)*math.cos(th)*math.sin(np.pi-la)+al*math.cos(np.pi-la)*math.cos(th)+rp*math.cos(th)
                    b=rs*math.cos(np.pi-la)*(math.cos(th))**2
                    c=rs**2+al**2+rp**2 - 2*rs*rp*math.sin(th)*math.sin(la)+ 2*al*rp*math.cos(np.pi-la)
                    e=-2*(al+rp*math.cos(np.pi-la))*rs*math.cos(th)
                    mu=abs(-rs+(al+rp*math.cos(np.pi-la))*math.cos(th)*math.cos(x)+rp*math.sin(th)*math.sin(la))/(c+e*math.cos(x))**(1/2)
                    if para==1:
                        lf=1-u*(1-mu)
                    if para==2:
                        lf=1-u1*(1-mu)-u2*(1-mu)**2
                    return abs(a-b*math.cos(x))*lf*mu/(c+e*math.cos(x))**(3/2)
                def integration(th,la): #First integral
                    return quad(function,-y3,y3,args=(th,la),**QUAD_KW)[0]   # MOD: QUAD_KW empty by default
                
                #Geometric Limits
                
                if la>0 and abs(la)<symp: 
                    ll=-math.acos((rs*(al-rp*math.cos(la))+rp*math.sin(la)*np.sqrt(al**2+rp**2-rs**2-2*al*rp*math.cos(la)))/(al**2+rp**2-2*al*rp*math.cos(la)))
                    ul=math.acos((rs*(al-rp*math.cos(la))-rp*math.sin(la)*np.sqrt(al**2+rp**2-rs**2-2*al*rp*math.cos(la)))/(al**2+rp**2-2*al*rp*math.cos(la)))
                    
                if la<0 and abs(la)<symp:
                    ll=-math.acos((rs*(al-rp*math.cos(la))+rp*math.sin(la)*np.sqrt(al**2+rp**2-rs**2-2*al*rp*math.cos(la)))/(al**2+rp**2-2*al*rp*math.cos(la)))
                    ul=math.acos((rs*(al-rp*math.cos(la))-rp*math.sin(la)*np.sqrt(al**2+rp**2-rs**2-2*al*rp*math.cos(la)))/(al**2+rp**2-2*al*rp*math.cos(la)))
                
                if abs(la)>=symp and la>0:
                    ll=-la+math.acos((al*math.cos(la)-rp)/rs)
                    ul=math.acos((rs*(al-rp*math.cos(la))-rp*math.sin(la)*np.sqrt(al**2+rp**2-rs**2-2*al*rp*math.cos(la)))/(al**2+rp**2-2*al*rp*math.cos(la)))
                    
                if abs(la)>=symp and la<0:
                    
                    ul=-la-math.acos((al*math.cos(la)-rp)/rs)
                    ll=-math.acos((rs*(al-rp*math.cos(la))+rp*math.sin(la)*np.sqrt(al**2+rp**2-rs**2-2*al*rp*math.cos(la)))/(al**2+rp**2-2*al*rp*math.cos(la)))
                
                if la==0:
                    ll=math.acos(rs/(al-rp))
                    ul=math.acos(rs/(al-rp))
                
                #Second integral
                
                value=quad(lambda th: integration(th,la),ll, ul,**QUAD_KW)[0]   # MOD: QUAD_KW empty by default
                la5=math.atan(al*math.tan(la)*math.cos(la)/(al*math.cos(la)-rp))
                if la5<0:
                    newp=la
                    newa.append(newp)
    
                value2=value*P/(4*np.pi*np.pi)
                final.append(value2/1000)
                
                old=P1*math.cos(la5)/(4*np.pi*(al**2+rp**2-2*al*rp*math.cos(la)))
                app=P1*np.exp(-0.7*la**2)/(4*np.pi*(al**2+rp**2-2*al*rp*math.cos(la)))
                if la>0 and la5<0:
                    old=0
                if la<0 and la5>0:
                    old=0
                oldfor.append(old/1000)
                approx.append(app/1000)
            
            err=abs(oldfor[NLAT//2]-final[NLAT//2])   # MOD: was index 250 (unused diagnostic)
            
           
            inverse.append(oldfor)
            average.append(final)
            aver=np.asarray(average)
            inve=np.asarray(inverse)
            inve1=np.mean(inve,axis=0)
            aver1=np.mean(aver,axis=0)
        
        maxlatitude=s1*57.3
        minlatitude=newa[0]*57.3
        tickarray2=np.array([-150,-130,-110,-90,-70,-50,-30,-10,10,30,50,70,90,110,130,150])
        plt.subplot(row,col,te)    
        plt.plot(la2,aver1,'b-',label="Geometric Model")
        plt.plot(la2,inve1,'r--', label="Inverse-square law")
        plt.xticks(tickarray2,fontsize=14)
        plt.yticks(fontsize=14)
        plt.axvline(x=maxlatitude,color='gray',linestyle='--',label='Terminator limit')
        plt.axvline(x=-maxlatitude,color='gray',linestyle='--')
        plt.axvline(x=symp*57.3,color='gray',linestyle='dashdot',label='Critical point of symmetry')
        plt.axvline(x=-symp*57.3,color='gray',linestyle='dashdot')
        plt.title("{0},[a={1},Rs={2},Teff={3},Rp={4}]".format(exoplanet,round(al1,4),round(rs1,2),fa1,round(rp1,2)),fontsize=16)
        plt.xlabel("Latitude",fontsize=16)
        plt.ylabel("Irradiance (kW/m^2)",fontsize=16)
        plt.legend(fontsize=16)
        plt.xlim(-130,130)
imagename=exoplanet+".jpg"
if num>1:
    imagename=input("Saving image..Enter image name and format:")
# MOD: export the plotted arrays (last planet of the run) and save the figure next to this script
tag=exoplanet.replace(" ","_")+"_u{:g}".format(u)+os.environ.get("INSTELLCA_TAG","")
np.savetxt(os.path.join(OUT_DIR,tag+"_instellca.txt"),np.c_[la2,aver1,inve1],
           header="InstellCa_3.0 (u={:g}, quad={}, nlat={}, {}): angle_from_substellar_deg  geometric_model_kW_m2  inverse_square_kW_m2".format(u,QUAD_KW or "scipy defaults",NLAT,plt.gca().get_title()))
imagename=os.path.join(OUT_DIR,tag+".png")
plt.savefig(imagename,dpi=300)
plt.show()  
print("The Terminator extends to ",round(s1*57.3,3), "degrees from the equator")

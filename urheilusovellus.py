import streamlit as st
import numpy as np
import folium 
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt
from scipy.signal import butter, filtfilt

locpath = "Location.csv"
accelpath = "Linear acceleration.csv"
locdf = pd.read_csv(locpath)
acceldf = pd.read_csv(accelpath)

locdf = locdf[locdf['Horizontal Accuracy (m)'] < 4]
locdf = locdf.reset_index(drop=True)

def haversine(lon1, lat1, lon2, lat2):
  lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * asin(sqrt(a))
  r = 6371
  return c * r

def butter_lowpass_filter(data, cutoff, fs, nyq, order):
  normal_cutoff = cutoff / nyq
  #Get the filter coefficients
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = filtfilt(b, a, data)
  return y

#Filtterien parametrit
T = acceldf['Time (s)'][len(acceldf['Time (s)']) - 1] - acceldf['Time (s)'][0] #Koko datan pituus
n = len(acceldf['Time (s)']) #Datapisteiden lukumäärä
fs = n/T #Näytteenottotaajuus (olettaen vakioksi)
nyq = fs/2 #Nyqvistin taajuus 
order = 3 #Kertaluku
cutoff = 1/(0.4) #Cut-off taajuus

#Kokeillaan suodatinta
filt_signal_array = butter_lowpass_filter(acceldf['Linear Acceleration x (m/s^2)'], cutoff, fs, nyq, order)
filt_signal = pd.concat([acceldf['Time (s)'], pd.Series(filt_signal_array, name = "Linear Acceleration x (m/s^2)")], axis=1)

#Lasketaan askeleiden määrä
jaksot = 0
for i in range(n-1):
  if filt_signal_array[i]/filt_signal_array[i+1] < 0: #True jos nollan ylitys, False jos ei ole
    jaksot = jaksot + 1

#Lasketaan kiihtyvyysdatan tehospektritiheys
noisy_signal = filt_signal['Linear Acceleration x (m/s^2)']
a = filt_signal['Time (s)']

#Lasketaan signaalin fourier-muunnos ja tehospektri
f = noisy_signal
N = len(f) #Datapisteiden määrä, eli f:n pituus
fourier = np.fft.fft(f, N) #Fourier-muunnos, sisältää tiedon signaalin f sisältämistä taajuuksista
psd = fourier * np.conj(fourier) / N #Tehospektri. Jokaisen rivin kompleksiluvut kerrotaan kompleksikonjugaatillaan
dt = np.max(a)/N

freq = np.fft.fftfreq(N, dt)
L = np.arange(1, np.floor(N/2), dtype='int') #Rajataan pois negatiiviset taajuudet ja nollataajuus tehospektristä
PSD = np.array([freq[L], psd[L].real])

#Alustetaan uudet sarakkeet
locdf['dist'] = np.zeros(len(locdf))
locdf['time_diff'] = np.zeros(len(locdf))

#Lasketaan keskinopeus ja kokonaismatka
for i in range(len(locdf)-1):
  locdf.loc[i,'dist'] = haversine(locdf['Longitude (°)'][i], locdf['Latitude (°)'][i], locdf['Longitude (°)'][i+1], locdf['Latitude (°)'][i+1])*10000
  locdf.loc[i,'time_diff'] = locdf['Time (s)'][i+1] - locdf['Time (s)'][i]
  locdf['velocity'] = locdf['dist'] / locdf['time_diff']
  locdf['tot_dist'] = np.cumsum(locdf['dist'])

askelpituus = (locdf['tot_dist'].max()/(jaksot/2))*10
keskinopeus = (locdf["Velocity (m/s)"].mean()/3.6)*10
laskettu_keskinopeus = (locdf["velocity"].mean()/3.6)
kokonaismatka = locdf["tot_dist"].max()/10

#Tehospektrin mukaan laskettu askelmäärä
p_max = np.argmax(PSD[1,:])
f_max = PSD[0,p_max]
t = np.max(acceldf['Time (s)'])
f_max = f_max*t

#Näytetään tulokset
st.title("Urheilusovellus")
st.write("Datan mukainen keskinopeus on :", round(keskinopeus, 1),'km/h')
st.write("Laskettu keskinopeus on :", round(laskettu_keskinopeus, 1),'km/h')
st.write("Laskettu kokonaismatka on :", round(kokonaismatka),'m')
st.write("Filtteröidyn signaalin mukaan laskettu askelten määrä on :", round(jaksot/2))
st.write("Fourier-analyysin perusteella laskettu askelmäärä on:", round(f_max))
st.write("Laskettu askelten pituus on :", int(askelpituus), 'cm')

#Filtteröidyn signaalin kuvaaja
st.subheader("Kuvaaja filtteröidystä signaalista", divider="gray")
st.write("Kuvaajasta näytetään vain aikaväli 30-100 sekuntia, jotta kuvaaja olisi selkeämpi.")
plt.figure(figsize=(15,5))
plt.plot(acceldf['Time (s)'],filt_signal)
plt.ylabel('Kiihtyvyys (m/s^2)')
plt.xlabel('Aika (s)')
plt.axis([30, 100, -10, 10])
plt.grid()
st.pyplot(plt)

#Kiihtyvyysdatan tehospektrikuvaaja
st.subheader("Kiihtyvyysdatan tehospektritiheys", divider="gray")
plt.figure(figsize=(15,5))
plt.plot(PSD[0,:], PSD[1,:])
plt.ylabel('Teho')
plt.xlabel('Taajuus')
plt.axis([0, 10, 0, 180000])
plt.grid()
st.pyplot(plt)

#Kartta
st.subheader("Karttakuva kuljetusta matkasta", divider="gray")
start_lat = locdf['Latitude (°)'].mean()
start_long = locdf['Longitude (°)'].mean()
map = folium.Map(location=[start_lat, start_long], zoom_start=17)

folium.PolyLine(locdf[['Latitude (°)', 'Longitude (°)']], color='blue').add_to(map)
st_map = st_folium(map, width=900, height=650)
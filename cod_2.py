import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt

"""             ######## Refencias ########
#cal: senal de calibración senoidal de 1 kHz a 1 Pa (94 dBSPL).
#ruido: senal de ruido de fondo.
#rms: valor eficaz de la senal de calibracion.
#f0: frecuencias centrales de las bandas de 1/3 de octava.
#Leq: vector que almacena el Leq de cada banda.
#leq: valor que almacena el valor de una banda.
                ###########################
"""

#importo la senal de calibracion y de ruido de fondo
cal, fs = sf.read('calibración_prueba.wav') #importo senal de calibracion
ruido, fs = sf.read('ruido_fondo_prueba.wav') #importo ruido de fondo


#1) Calculo el valor RMS de la senal de calibracion
rms = np.sqrt(sum(cal**2)/len(cal)) 

#2) Aplico la regla de tres simple para pasar de 
# tensión equivalente (eV) a Pascales (Pa)
ruido = ruido/rms

#3) Filtro la senal ruido por tercio de octava.
# Estas son las frecuencias centrales normalizadas que figuran en la normativa
# ANSIS XXX de cada banda de tercio de octava

f0 = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 
          800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
          10000, 12500, 16000]

N = np.arange(14,43,1) #Leer pag. 5 de ANSI
K = N - 30
fr = 1000
fm = fr*(2**(K/3))

# para calcular las frecuencias superior e inferior hay que aplicar
# f1 = 2^(1/6) * fcentral
# f2 = 2^(-1/6) * fcentral
# para filtrarlo se utiliza un butterworth pasabanda de 6to orden, indicado
# por la normativa.

Leq=[] 

for i in range(0,len(f0)):
    Wn = ((f0[i]*(2**(-1/6)/(0.5*fs))),f0[i]*(2**(1/6)/(0.5*fs)))
    sos = signal.butter(6,Wn,btype='bandpass',output='sos',analog=True)
    a,b = signal.butter(6,Wn,btype='bandpass',output='ba',analog=True)
    w, h = signal.freqs(a,b)
    #plt.semilogx(w, 20*np.log10(abs(h)))
    ruido_fil = signal.sosfilt(sos, ruido)
    
    rms_banda=np.sqrt(sum(ruido_fil**2)/len(ruido_fil))
    leq = 20*np.log10(rms_banda/20e-6)
    print(rms_banda)
    Leq.append(leq)
    

print(Leq)
    
    
    
#4) Ploteo del gráfico de barras

xlabels = ['25' ,'31.5', '40','50','63','80','100','125','160','200','250',
           '315','400','500','630','800','1k','1.25k','1.6k','2k','2.5k',
           '3.15k','4k','5k','6.3k','8k','10k','12.5k','16k']

plt.bar(f0,Leq)
plt.tight_layout()
# plt.xlim(20,20000)
# plt.ylim(-20, 2.5)
# plt.xlabel(r'$Frecuencia\ [Hz]$', fontsize=12)
# plt.ylabel(r'$Magnitud\ [dB]$', fontsize=13)

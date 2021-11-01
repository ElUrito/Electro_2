import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt

               ######## Refencias ########
#cal: senal de calibración senoidal de 1 kHz a 1 Pa (94 dBSPL).
#ruido: senal de ruido de fondo.
#rms: valor eficaz de la senal de calibracion.
#f0: frecuencias centrales de las bandas de 1/3 de octava.
#Leq: vector que almacena el Leq de cada banda.
#leq: valor que almacena el valor de una banda.
                ###########################
                
#importo la senal de calibracion y de ruido de fondo
cal, fs = sf.read('calibracion_prueba.wav') #importo senal de calibracion
ruido, fs = sf.read('ruido_fondo_prueba.wav') #importo ruido de fondo


#1) Calculo el valor RMS de la senal de calibracion
rms = np.sqrt(sum(cal**2)/len(cal)) 

#2) Aplico la regla de tres simple para pasar de 
# tensión equivalente (eV) a Pascales (Pa)
ruido = ruido/rms

#3) Filtro la senal ruido por tercio de octava.
# Estas son las frecuencias centrales normalizadas que 
# figuran en la normativa ANSI S1.11-1986 por cada 
# banda de tercio de octava

f0 = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 
          800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
          10000, 12500, 16000]


# para calcular las frecuencias superior e inferior hay que aplicar
# f1 = 2^(1/6) * fm
# f2 = 2^(1/6) * fm

b = 1/3 
N = np.arange(14,44,1) 
K = N - 30 
fr = 1000  #la frecuencia de referencia es de 1 kHz para audio

fmax = []
fmin = []
Fm = []

for i in range(len(N)):
    fm = fr*(2**(K[i]*b)) # ¿Por qué?
    Fm.append(fm)
    f1 = (fm*(2**(-b/2)))/(0.5*fs)
    fmin.append(f1)
    f2 = (fm*(2**(b/2)))/(0.5*fs)
    fmax.append(f2)
    
print(fmin)
print(fmax)
print(Fm)


# El tipo de filtro que se indica en la normativa es un 
# butterworth pasabanda de 6to orden. Luego se calcula 
# el valor rms de la senal por banda y finalmente se 
# expresa este valor en dB SPL

Wn = [fmin,fmax]
SPL = []

for i in range(0,len(N)):
    sos = signal.butter(6,(Wn[0][i],Wn[1][i]),btype='bandpass',output='sos')
    a,b = signal.butter(6,(Wn[0][i],Wn[1][i]),btype='bandpass',analog=True,output='ba')
    w, h = signal.freqs(a,b) 
    plt.semilogx(w*fs/2, 20*np.log10(abs(h))) #¿Por qué?
    ruido_fil = signal.sosfilt(sos, ruido)
    
    rms_banda=np.sqrt(sum(ruido_fil**2)/len(ruido_fil))
    spl = 20*np.log10(rms_banda/20e-6)
    SPL.append(spl)

# hago un vector de strings de las frecuencias 
# centrales normalizadas para el ploteo.
xlabels = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250', '315', '400', '500', '630', 
          '800', '1000', '1250', '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000',
          '10000', '12500','16000']

plt.xticks(f0, xlabels, rotation=70)
plt.xlim(20,20000)
plt.ylim(-10, 2.5)
plt.xlabel(r'$Frecuencia\ (Hz)$', fontsize=12)
plt.ylabel(r'$Magnitud\ (dB)$', fontsize=13)
plt.grid()
plt.tight_layout()
plt.savefig('rta_filtros.png')    

# genero un vector de igual largo que el de
# las bandas normalizadas
freqs = np.arange(0,len(N)-1,1)

plt.figure()
plt.bar(range(len(Fm)),SPL)
plt.xticks(freqs,xlabels,rotation=70)
plt.ylabel(r'$L_{eq} \ (dBSPL)$', fontsize=13)
plt.xlabel(r'$Frecuencia \ (Hz)$', fontsize=12)
plt.tight_layout()
plt.savefig('ruido_tercios.png')
plt.show()

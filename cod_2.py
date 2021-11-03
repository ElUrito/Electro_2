# Importación de módulos
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt


cal, fs = sf.read('Audios prueba/calibración.wav')  # importo senal de calibracion y su frecuencia de muestreo
ruido, _ = sf.read('Audios prueba/senal.wav')  # importo ruido de fondo y su frecuencia de muestreo

cal2, fs2 = sf.read('Archivos/med_calibracion.wav')  # importo senal de calibracion y su frecuencia de muestreo
ruido2, _ = sf.read('Archivos/med_ruido.wav')  # importo ruido de fondo y su frecuencia de muestreo

print(cal)
print(cal2)

rms = np.sqrt(sum(cal**2)/len(cal))   # calculo valor RMS de señal calibracion
ruido = ruido/rms  # paso de tensión equivalente a pascales


b = 1/3
N = np.arange(14, 44, 1)  # numero de banda (ver tabla 1, pag 5 de ANSI)
K = N - 30
fr = 1000  # la frecuencia de referencia es de 1 kHz para audio

fmax = []  # vector que acumula las frecuencias superirores de cada banda
fmin = []  # vector que acumula las frecuencias inferiores de cada banda
Fm = []    # vector que acumula el nivel SPL de cada banda

for i in range(len(N)):
    fm = fr*(2**(K[i]*b))  # genera las frecuencias centrales normalizadas
    Fm.append(fm)
    f1 = (fm*(2**(-b/2)))/(0.5*fs)  # genera las frecuencias inferiores (f1 en ANSI)
    fmin.append(f1)
    f2 = (fm*(2**(b/2)))/(0.5*fs)  # genera las frecuencias inferiores (f1 en ANSI)
    fmax.append(f2)
   

Wn = [fmin, fmax]
SPL = []  # vector que acumula el nivel SPL de cada banda


for i in range(0, len(N)):
    sos = signal.butter(6, (Wn[0][i], Wn[1][i]), btype='bandpass', output='sos')  # se obtiene un vector
    # con los coeficientes a_i y b_i de H(z)
    a, b = signal.butter(6, (Wn[0][i], Wn[1][i]), btype='bandpass', analog=True, output='ba')  # se obtienen
    # a y b de H(z) = A(z)/B(z)
    w, h = signal.freqs(a, b)  # se obtiene la respuesta en frecuencia a partir de los coeficientes a y b
    plt.semilogx(w*fs/2, 20*np.log10(abs(h)), linewidth=3)  # se plotean la magnitud de los filtros por banda
    ruido_fil = signal.sosfilt(sos, ruido)  # se filtra el ruido de fondo
    
    rms_banda = np.sqrt(sum(ruido_fil**2)/len(ruido_fil))  # se obtiene el valor rms del ruido filtrado en esta banda
    spl = 20*np.log10(rms_banda/20e-6)  # se pasa a dB SPL el valor en eV
    SPL.append(spl)  # se acumula en el vector SPL


fcentral = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 
          800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
          10000, 12500, 16000, 20000]

xlabels = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250', '315', '400', '500', '630', 
          '800', '1000', '1250', '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000',
          '10000', '12500', '16000', '20000']


# plt.figure(figsize=(18, 5))
plt.xticks(fcentral, xlabels, rotation=70, fontsize=20)
plt.xlim(20, 25000)
plt.ylim(-10, 2.5)
plt.xlabel(r'$Frecuencia\ (Hz)$', fontsize=25)
plt.ylabel(r'$Magnitud\ (dB)$', fontsize=25)
plt.tick_params(axis='y', labelsize=20)
plt.grid()
# plt.savefig('rta_filtros.png')

bandas = np.arange(0, len(N), 1)

# plt.figure(figsize=(18, 5))
plt.tick_params(axis='y', labelsize=20)
plt.bar(range(len(Fm)), SPL)
plt.xticks(bandas, xlabels, rotation=70, fontsize=20)
plt.ylabel(r'$L_{p} \ (dBSPL)$', fontsize=25)
plt.xlabel(r'$Frecuencia \ (Hz)$', fontsize=25)
# plt.savefig('ruido_tercios.png')
plt.grid()
plt.show()


"""
1) Calculo el valor RMS de la senal de calibracion
2) Aplico la regla de tres simple para pasar de tensión equivalente (eV) a Pascales (Pa)

3) Filtro la senal ruido por tercio de octava. Estas son las frecuencias 
centrales normalizadas que figuran en la normativa ANSI S1.11-1986 por cada 
banda de tercio de octava.
[25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 
800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
10000, 12500, 16000]

para calcular las frecuencias superior e inferior hay que aplicar
f1 = 2^(-1/6) * fm = 2^(-1/6) * (fr* 2^(K*b)) 
f2 = 2^(1/6) * fm = 2^(1/6) * (fr* 2^(K*b))

El tipo de filtro que se indica en la normativa es un butterworth
pasabanda de 6to orden. Luego se calcula el valor rms de la senal
por banda y finalmente se expresa este valor en dB SPL.


#5) Grafico el ruido y la respuesta en frecuencia de cada filtro hago un vector
# de strings de las frecuencias centrales normalizadas para el ploteo.

"""
# Importación de librerías
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from cod_1 import matriz_impulsos


# Función

def mag_real(senal, orientacion='h', graficar='', guardar_img=''):
    eje = 'horizontal'
    if orientacion == 'v' or orientacion == 'V':
        eje = 'vertical'

    # Calibración
    cal, fs = sf.read('Archivos/med_calibracion.wav')  # importo senal de calibracion y su frecuencia de muestreo
    rms = np.sqrt(sum(cal ** 2) / len(cal))  # calculo valor RMS de señal calibracion
    senal_sel = np.zeros([4, 195])
    senal_sel[0] = senal[36]  # 0°
    senal_sel[1] = senal[39]  # 15°
    senal_sel[2] = senal[42]  # 30°
    senal_sel[3] = senal[45]  # 45°
    senal_cal = senal_sel / rms  # paso de tensión equivalente a pascales

    # FFT
    senal_fft = abs(np.fft.rfft(senal_cal))
    senal_spl = 20 * np.log10(senal_fft / 0.00002)

    f = np.linspace(0, fs / 2, len(senal_spl[0]))

    # Gráfico contorno
    labels = ['0°', '15°', '30°', '45°']
    plt.figure(figsize=[8.5, 6.5])
    for i in range(4):
        plt.semilogx(f, senal_spl[i], label=labels[i])
        plt.xlim(250, 25000)
    ticksx = [250, 500, 1000, 2000, 4000, 8000, 16000]
    plt.xticks(ticks=ticksx, labels=ticksx, rotation=0)
    plt.ylabel(r'Nivel de presión sonora [dB SPL]')
    plt.xlabel(r'Frecuencia [Hz]')
    plt.ylim(70, np.max(senal_spl) * 1.05)
    plt.grid(linewidth=1, which='both')
    plt.legend(framealpha=1)
    plt.tight_layout()
    if guardar_img == 'si' or guardar_img == 'SI' or guardar_img == 'Si':
        plt.savefig(f'Rta del sist {eje}')
    if graficar == 'si' or graficar == 'SI' or graficar == 'Si':
        plt.show()


def graf_polar(frecuencias, senal, orientacion='h', graficar='', guardar_img=''):
    eje = 'horizontal'
    if orientacion == 'v' or orientacion == 'V':
        eje = 'vertical'

    # Gráfico polar
    grados = np.linspace(0, 180, 37, dtype=int)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    theta = np.zeros([73])
    theta = np.pi * np.hstack((-1 * np.flip(grados), grados)) / 180
    theta = np.delete(theta, 37)

    freqs = [1000, 2000, 4000, 8000, 16000]
    labels = ['1 kHz', '2 kHz', '4 kHz', '8 kHz', '16 kHz']

    mat_polar = np.zeros([5, 73])
    mat_polar[0] = senal[:, 23]
    mat_polar[1] = senal[:, 43]
    mat_polar[2] = senal[:, 86]
    mat_polar[3] = senal[:, 171]
    mat_polar[4] = senal[:, 342]
    ax.plot(theta, mat_polar[0], label=labels[0])
    ax.plot(theta, mat_polar[1], label=labels[1])
    ax.plot(theta, mat_polar[2], label=labels[2])
    ax.plot(theta, mat_polar[3], label=labels[3])
    ax.plot(theta, mat_polar[4], label=labels[4])

    ax.set_rlim((-55, 0))
    ax.set_rticks(np.arange(-50, 0, 10))
    ax.set_rlabel_position(0)
    ax.tick_params(axis='x', which='both', labelsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
    ax.set_title(f'Patrón polar {eje}', fontsize=13, pad=15)
    plt.tight_layout()
    if guardar_img == 'si' or guardar_img == 'SI' or guardar_img == 'Si':
        plt.savefig(f'polar_{eje}.png')
    if graficar == 'si' or graficar == 'SI' or graficar == 'Si':
        plt.show()


# Ejecución de función
"""
Para ejecutar esta función se debe primeramente llamar a la función "matriz_impulso()" del cod_1.py, donde
se cargan los impulsos, se da formato y se extrae en forma de matriz, siendo el segundo valor que retorna
dicha función. Luego se ingresa dicha matriz a la función de este script (mag_real()), que se encarga de
calibrarla y graficarla con magnitud real en dB SPL.

Para ejecutar el código y ver el gráfico, hacerlo de la siguiente manera, habiendo conseguido la matriz primero:
mag_real(senal, graficar='si')

Para guardar dicha imagen, se debe hacer de la siguiente manera:
mag_real(senal, guardar_img='si')

Se puede graficar y guardar la imagen en simultáneo de la siguiente manera:
mag_real(senal, graficar='si', guardar_img='si')
"""
_, senal, f, senal_fft = matriz_impulsos()
# mag_real(senal, graficar='si')
graf_polar(f, senal_fft, guardar_img='si')

# _, senal2, f2, senal_fft2 = matriz_impulsos(orientacion='v')
# graf_polar(f2, senal_fft2, guardar_img='si', orientacion='v')

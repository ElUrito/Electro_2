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
    senal_sel[0] = senal[36]
    senal_sel[1] = senal[39]
    senal_sel[2] = senal[42]
    senal_sel[3] = senal[45]
    senal_cal = senal_sel / rms  # paso de tensión equivalente a pascales

    # FFT
    senal_fft = abs(np.fft.rfft(senal_cal))
    senal_spl = 20 * np.log10(senal_fft / 0.00002)

    f = np.linspace(0, fs / 2, len(senal_spl[0]))

    # Gráfico
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


# Ejecución de función
_, senal, _, _ = matriz_impulsos()
mag_real(senal, graficar='si')

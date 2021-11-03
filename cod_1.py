# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib import cm


# Funciones


def matriz_impulsos(orientacion="H"):
    """
    Esta función obtiene los distintos impulsos de los archivos .csv, los convierte en un vector "tiempo"
    y una matriz "impulsos" compuesta por vectores que son los valores a distintos ángulos. También aplica
    una FFT a dicha matriz, creando otra "imp_fft" y su vector del eje frecuencial.
    :param orientacion: Para eje horizontal, "H" o "h" (predeterminado). Para eje vertical, "V" o "v".
    :return: Guarda en la carpeta una imagen de contorno de formato .png.
    """

    # Definición de variables
    global eje
    array_conteo = np.linspace(0, 180, num=37, dtype=int)  # Vector de ángulos
    impulsos = np.array(np.zeros([73, 195]), dtype=object)  # Vector vacío para impulsos
    imp_fft = np.array(np.zeros([73, 513]), dtype=object)  # Vector vacío para impulsos pasados por fft
    tiempo = np.array([0])  # Vector vacío para tiempo
    N = 1024
    for i in range(len(array_conteo)):  # Elige eje horizontal o vertical. Por defecto, horizontal
        if orientacion == "H" or orientacion == "h":
            eje = 'horizontal'
            archivo = pd.read_csv("Archivos/de620me90_hor_deg{:.0f}.csv".format(array_conteo[i]), header=None)
        elif orientacion == "V" or orientacion == "v":
            eje = 'vertical'
            archivo = pd.read_csv("Archivos/de620me90_ver_deg{:.0f}.csv".format(array_conteo[i]), header=None)
        else:
            print('Para eje horizontal, orientacion="H"; para vertical, orientacion="V"')
            break
        archivo = archivo.drop([0, 1, 2])  # Elimina las primeras 3 filas que sólo tienen encabezados
        archivo2 = archivo.to_numpy()
        archivo2 = archivo2.astype(float)
        archivo3 = archivo2[293:488, 1]  # Ventaneo eligiendo sólo las muestras que quiero
        # (desde 6.125 ms durante 4.063 ms)
        if i == 0:
            tiempo = archivo2[294:489, 0]  # Definición vector tiempo con mismo ventaneo
            impulsos[36] = archivo3  # Definición matriz "impulsos" para ángulo 0
            imp_fft_ceros = np.hstack([archivo3, np.zeros(415)])
            imp_fft[36] = 20 * np.log10(abs(np.fft.rfft(imp_fft_ceros, N)))

        else:
            impulsos[i + 36] = archivo3  # Definición matriz "impulsos" para ángulos positivos
            impulsos[36 - i] = archivo3  # Definición matriz "impulsos" para ángulos negativos
            imp_fft_ceros = np.hstack([archivo3, np.zeros(415)])
            imp_fft[i + 36] = 20 * np.log10(abs(np.fft.rfft(imp_fft_ceros, N)))-imp_fft[36]
            imp_fft[36 - i] = 20 * np.log10(abs(np.fft.rfft(imp_fft_ceros, N)))-imp_fft[36]
        del archivo
        del archivo2
        del archivo3
    f = np.linspace(0, 24000, len(imp_fft[0]))
    tiempo = tiempo*1000
    imp_fft[36] = imp_fft[36]-imp_fft[36]

    plt.figure()
    contorno_x = f
    contorno_y = np.linspace(-180, 180, 73, dtype=int)
    levels = [-33, -30, -27, -24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6]
    plt.contour(contorno_x, contorno_y, imp_fft, levels=[-6])
    plt.contourf(contorno_x, contorno_y, imp_fft, levels, cmap='jet')
    plt.gca().set_xscale('log')
    ticksx = [500, 1000, 2000, 5000, 10000, 20000]
    ticksy = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
    plt.xticks(ticksx, ticksx)
    plt.xlim(250, 20000)
    plt.yticks(ticksy, ticksy)
    cbar = plt.colorbar(drawedges='True')
    cbar.set_label('Nivel relativo [dB]', fontsize=12)
    plt.title(f'Gráfico de contorno {orientacion} DE620', fontsize=13)
    plt.xlabel('Frecuencia [Hz]', fontsize=12)
    plt.ylabel('Ángulo [°]', fontsize=12)
    plt.grid()
    plt.savefig(f'Contorno eje {eje} DE620-8 con ME90.png')

    # return tiempo, impulsos, freq, imp_fft


matriz_impulsos()
matriz_impulsos(orientacion='v')


"""
Ventaneo: 195 muestras
R = 48000 HZ / 195 muestras = 246.15 Hz frec mínima
"""
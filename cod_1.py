# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Funciones

def graficos(frecuencias, imp_fft, graficar='Si', guardar_img='', eje='horizontal'):
    # Gráficos
    plt.figure()
    angulos = np.linspace(-180, 180, 73, dtype=int)
    levels = [-33, -30, -27, -24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6]
    plt.contour(frecuencias, angulos, imp_fft, levels=[-6])
    plt.contourf(frecuencias, angulos, imp_fft, levels, cmap='jet')
    plt.gca().set_xscale('log')
    ticksx = [500, 1000, 2000, 5000, 10000, 20000]
    ticksy = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
    plt.xticks(ticksx, ticksx)
    plt.xlim(250, 20000)
    plt.yticks(ticksy, ticksy)
    cbar = plt.colorbar(drawedges='True')
    cbar.set_label('Nivel relativo [dB]', fontsize=12)
    plt.title(f'Gráfico de contorno {eje} DE620', fontsize=13)
    plt.xlabel('Frecuencia [Hz]', fontsize=12)
    plt.ylabel('Ángulo [°]', fontsize=12)
    plt.grid()
    if guardar_img == "si" or guardar_img == "Si" or guardar_img == "SI":
        plt.savefig(f'Contorno eje {eje} DE620-8 con ME90.png')
    if graficar == "si" or graficar == "Si" or graficar == "SI":
        plt.show()


def matriz_impulsos(orientacion="H", graficar='Si', guardar_img=''):
    """
    Esta función obtiene los distintos impulsos de los archivos .csv, previamente exportados desde el ARTA
    y los convierte en un vector "tiempo" y una matriz "impulsos" compuesta por vectores que son los valores
    a distintos ángulos. También aplica una FFT a dicha matriz, creando otra "imp_fft" y su vector del eje
    frecuencial.
    :param orientacion: Para eje horizontal, "H" o "h" (predeterminado). Para eje vertical, "V" o "v".
    :param graficar: Poner 'Si' si desea graficar en la ejecución del código.
    :param guardar_img: Poner 'Si' si desea guardar la imagen en la carpeta.
    :return: Vector eje temporal, matriz de impulsos, vector eje frecuencial, matriz impulsos pasados por FFT.
    """

    # Definición de variables
    eje = 'horizontal'
    array_conteo = np.linspace(0, 180, num=37, dtype=int)  # Vector de ángulos
    impulsos = np.array(np.zeros([73, 195]), dtype=object)  # Vector vacío para impulsos
    imp_fft = np.array(np.zeros([73, 513]), dtype=object)  # Vector vacío para impulsos pasados por fft
    tiempo = np.array([0])  # Vector vacío para tiempo
    N = 1024
    for i in range(len(array_conteo)):  # Elige eje horizontal o vertical. Por defecto, horizontal
        if orientacion == "H" or orientacion == "h":
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
        # Ventaneo: 195 muestras; R = 48000 HZ / 195 muestras = 246.15 Hz frec mínima
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
    frecuencias = np.linspace(0, 24000, len(imp_fft[0]))
    tiempo = tiempo*1000  # Para pasar a milisegundos
    imp_fft[36] = imp_fft[36]-imp_fft[36]  # Normalizo a valores de 0°

    if __name__ == '__main__':
        graficos(frecuencias, imp_fft, graficar, guardar_img, eje)

    return tiempo, impulsos, frecuencias, imp_fft

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
Para ejecutar la función se puede hacer sencillamente llamando a la función sin ningún parámetro agregado.
Por defecto lee las mediciones del eje horizontal, no grafica ni guarda imagen. Para elegir el eje vertical, 
llamar a la función de la siguiente manera:
matriz_impulsos(orientacion='v')

Para graficar, hacerlo de la siguiente manera:
matrtiz_impulsos(graficar='si')

Para guardar imagen en la carpeta donde se está ejecutando el script, hacerlo de la siguiente manera:
matriz_impulsos(guardar_img='si')

También se pueden obtener los vectores temporal y frecuencial, así como también la matriz de impulsos
e impulsos transformados por Fourier de la siguiente manera:
vec_tiempo, mat_impulsos, vec_frecuencias, mat_imp_fft = matriz_impulsos()

El funcionamiento de graf_polar es análogo.
"""

matriz_impulsos(orientacion='v', graficar='si')
# matriz_impulsos(orientacion='v', guardar_img='si')

# graf_polar(f2, senal_fft2, guardar_img='si', orientacion='v')
# graf_polar(f, senal_fft, graficar='si', guardar_img='si')



# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def matriz_impulsos(orientacion="H"):
    """
    Esta función obtiene los distintos impulsos de los archivos .csv y convierte en un vector "tiempo" y una "impulsos"
    :param orientacion: Para eje horizontal, "H" o "h". Para eje vertical, "V" o "v".
    Eje horizontal como predeterminado.
    :return: Devuelve un vector "tiempo" para graficar como eje horizontal y una matriz "impulsos" con los
    valores a distintas direcciones (con pasos de 5°).
    """

    # Definición de variables
    array_conteo = np.linspace(0, 180, num=37, dtype=int)  # Vector de ángulos
    impulsos = np.array(np.zeros(73), dtype=object)  # Vector vacío para impulsos
    tiempo = np.array([0])  # Vector vacío para tiempo
    for i in range(len(array_conteo)):  # Elige eje horizontal o vertical. Por defecto, horizontal
        if orientacion == "H" or orientacion == "h":
            archivo = pd.read_csv("Archivos/de620me90_hor_deg{:.0f}.csv".format(array_conteo[i]), header=None)
        elif orientacion == "V" or orientacion == "v":
            archivo = pd.read_csv("Archivos/de620me90_ver_deg{:.0f}.csv".format(array_conteo[i]), header=None)
        else:
            print('Para eje horizontal, orientacion="H"; para vertical, orientacion="V"')
            break
        archivo = archivo.drop([0, 1, 2])  # Elimina las primeras 3 filas que sólo tienen encabezados
        archivo2 = archivo.to_numpy()
        archivo2 = archivo2.astype(float)
        archivo3 = archivo2[294:489, 1]  # Ventaneo eligiendo sólo las muestras que quiero
        # (desde 6.125 ms durante 4.063 ms)
        if (i == 0):
            tiempo = archivo2[294:489, 0]  # Definición vector tiempo con mismo ventaneo
            impulsos[36] = archivo3  # Definición matriz "impulsos" para ángulo 0
        else:
            impulsos[i + 36] = archivo3  # Definición matriz "impulsos" para ángulos positivos
            impulsos[36 - i] = archivo3  # Definición matriz "impulsos" para ángulos negativos
        del (archivo)
        del (archivo2)
        del (archivo3)
    return tiempo, impulsos

def fft(matriz):
    mat_fft = np.zeros(len(matriz))
    for i in range(len(matriz)):
        mat_fft[i] = np.fft.rfft(matriz[i])
    matriz_fft = 20*np.log10(mat_fft)
    freq = np.linspace(0, 24000, len(mat_fft))
    return freq, matriz_fft


tiempo, impulsos = matriz_impulsos()

frec, vector = fft(impulsos)
print(vector)

# Gráfico
# plt.subplot(221)
# plt.plot(frec, vector[0])
# plt.subplot(222)
# plt.plot(frec, vector[-1])
# plt.subplot(223)
# plt.plot(frec, vector[1])
# plt.subplot(224)
# plt.plot(frec, vector[-2])
# plt.show()

"""
NO FUNCIONA FFT
En otra función hacer rfft y normalizar al valor en 0°. Hacer función gráfico.
"""
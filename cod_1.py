# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Funciones


def matriz_impulsos(orientacion="H"):
    """
    Esta función obtiene los distintos impulsos de los archivos .csv y convierte en un vector "tiempo" y una "impulsos"
    :param orientacion: Para eje horizontal, "H" o "h". Para eje vertical, "V" o "v".
    Eje horizontal como predeterminado.
    :return: Devuelve un vector "tiempo" para graficar como eje horizontal, una matriz "impulsos" con los
    valores a distintas direcciones (con pasos de 5°), un vector con el eje frecuencial de la fft y una
    matriz de fft.
    """

    # Definición de variables
    array_conteo = np.linspace(0, 180, num=37, dtype=int)  # Vector de ángulos
    impulsos = np.array(np.zeros(73), dtype=object)  # Vector vacío para impulsos
    nuevo_imp = np.zeros(73)
    imp_fft = np.array(np.zeros(73), dtype=object)  # Vector vacío para impulsos pasados por fft
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
        if i == 0:
            tiempo = archivo2[294:489, 0]  # Definición vector tiempo con mismo ventaneo
            impulsos[36] = archivo3  # Definición matriz "impulsos" para ángulo 0
            imp_fft[36] = 20 * np.log10(abs(np.fft.rfft(archivo3)))
        else:
            impulsos[i + 36] = archivo3  # Definición matriz "impulsos" para ángulos positivos
            impulsos[36 - i] = archivo3  # Definición matriz "impulsos" para ángulos negativos
            imp_fft[i+36] = 20 * np.log10(abs(np.fft.rfft(archivo3)))
            imp_fft[36-i] = 20 * np.log10(abs(np.fft.rfft(archivo3)))
        del archivo
        del archivo2
        del archivo3
    freq = np.linspace(0, 24000, len(imp_fft[0]))
    tiempo = tiempo*1000
    # imp_fft[36] = imp_fft[36]-imp_fft[36]
    return tiempo, impulsos, freq, imp_fft


def grafico(t, mat_t, f, mat_f):

    contorno_x = f
    contorno_y = np.linspace(-180, 180, 73, dtype=int)
    print(type(contorno_y))
    print(type(contorno_x))
    levels = [-33, -30, -27, -24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6]
    X, Y = np.meshgrid(contorno_x, contorno_y)
    plt.contourf(X, Y, mat_f, levels, cmap='jet', extend='min')

    # plt.figure(1)
    # plt.subplot(121)
    # plt.plot(t, mat_t[35])
    # plt.subplot(122)
    # plt.semilogx(f, mat_f[35])
    #
    # plt.figure(2)
    # plt.subplot(121)
    # plt.plot(t, mat_t[36])
    # plt.subplot(122)
    # plt.semilogx(f, mat_f[36])
    #
    # plt.figure(3)
    # plt.subplot(121)
    # plt.plot(t, mat_t[37])
    # plt.subplot(122)
    # plt.semilogx(f, mat_f[37])
    plt.show()


tiempo, impulsos, frecuencia, imp_fft = matriz_impulsos()
print(type(imp_fft))

grafico(tiempo, impulsos, frecuencia, imp_fft)


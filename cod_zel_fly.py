import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy as sc


# %%

def contorno(fs, t_ref, archivo, eje='h', elemento=''):

    if eje == 'v':
        orientacion = 'vertical'
    elif eje == 'h':
        orientacion = 'horizontal'

    lim_sup = round(fs * t_ref + 4)
    grados = np.linspace(0, 180, 37, dtype=int)
    df = pd.read_excel(archivo)
    h = df[grados][4:lim_sup]
    h = np.array(h)
    h = np.transpose(h)

    ############ FFT #############

    v_fft = np.hstack([h, np.zeros((37, fs - lim_sup))])
    v_fft = np.fft.rfft(v_fft)
    v_fft_log = 20 * np.log10(abs(v_fft))
    matriz_polar = v_fft_log - v_fft_log[0, :]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    theta = np.pi * np.hstack((-1 * np.flip(grados), grados)) / 180
    freqs = [1000, 2000, 4000, 8000, 12000, 16000]
    labels = ['1 kHz', '2 kHz', '4 kHz', '8 kHz', '12 kHz', '16 kHz']

    for i, f in enumerate(freqs):
        x = np.hstack([np.flip(matriz_polar[:, f]),
                       matriz_polar[:, f]])

        ax.plot(theta, x, label=labels[i])

    ax.set_rlim((-55, 0))
    ax.set_rticks(np.arange(-50, 0, 10))
    ax.set_rlabel_position(0)
    ax.tick_params(axis='x', which='both', labelsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
    ax.set_title(f'Patrón polar {orientacion} ({elemento})', fontsize=13,
                 pad=15)
    plt.tight_layout()
    # plt.savefig(f'polar_{eje}_{elemento}.png')

    ########## CONTORNO ##################

    matriz_contorno = np.vstack((np.flip(np.delete(matriz_polar, 0, 0), axis=0),
                                 matriz_polar))

    plt.figure()
    contorno_y = np.linspace(-180, 180, 73, dtype=int)
    contorno_x = np.linspace(0, fs / 2, len(matriz_contorno[0, :]))
    levels = [-33, -30, -27, -24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6]
    X, Y = np.meshgrid(contorno_x, contorno_y)
    plt.contour(X, Y, matriz_contorno, levels=[-6])
    plt.contourf(X, Y, matriz_contorno, levels, cmap='jet', extend='min')
    plt.gca().set_xscale('log')
    ticksx = [500, 1000, 2000, 5000, 10000, 20000]
    ticksy = [-180, -90, 0, 90, 180]
    plt.xticks(ticksx, ticksx)
    plt.xlim(500, 20000)
    plt.yticks(ticksy, ticksy)
    cbar = plt.colorbar(drawedges='True')
    cbar.set_label('Nivel relativo [dB]', fontsize=12)
    plt.title(f'Gráfico de contorno {orientacion} ({elemento})', fontsize=13)
    plt.xlabel('Frecuencia [Hz]', fontsize=12)
    plt.ylabel('Ángulo [°]', fontsize=12)
    # plt.savefig(f'contorno_{eje}_{elemento}.png')
    # plt.show()


fs = 48000
t_ref = 0.0108
archivo_H = 'Bocina_ME20_0_180_H.xlsx'
# archivo_V = 'Bocina_ME20_0_180_V.xlsx'

contorno(fs, t_ref, archivo_H, eje='h', elemento='ME20')
# contorno(fs, t_ref, archivo_V, eje='v', elemento='ME20')
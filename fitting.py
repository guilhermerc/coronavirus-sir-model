#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Este código contém o ajuste de parâmetros do modelo SIR aos dados oficiais
    do total de pessoas infectadas por COVID-19 no Brasil até 23/06/20.
    Autor: Guilherme Ricioli Cruz <guilherme.riciolic@gmail.com>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint

# Lendo os dados oficiais do total de pessoas infectadas pela COVID-19 até 23/06
# (fonte: Ministério da Saúde <https://covid.saude.gov.br/>)
# ##############################################################################
data = np.array(pd.read_csv(r'data/acumulados-23-06.csv', header=None))
data = np.ndarray.flatten(data)
t = np.arange(len(data))

M = 211696880.0 # População brasileira às 19:35:40 de 25/06/2020 (fonte: IBGE
                # <https://www.ibge.gov.br/apps/populacao/projecao/box_popclock.php>)

# Definindo o sistema de EDOs do modelo SIR
# ##############################################################################
def system(x, t, gamma, alpha):
    s, i, a = x
    dxdt = [-gamma*(s/M)*i, gamma*(s/M)*i - alpha*i, gamma*(s/M)*i]
    return dxdt

# Definindo a função que retorna a solução numérica do sistema de EDOs do modelo
# SIR
# ##############################################################################
def solver(t, gamma, alpha, s_0, i_0, a_0):
    sol = odeint(system, [s_0, i_0, a_0], t, args=(gamma, alpha))
    return sol[:, 2] # Somente a[k] nos interessa

# Ajustando os parâmetros gamma, alpham s_0, i_0 e a_0 do modelo SIR
# ##############################################################################
popt, pcov = curve_fit(solver, t, data, bounds=([0.0, 0.0, 0.0, 0.0, 0.0], [50.0, 1.0, M, M, M]))

# Plotando os dados oficiais e a curva obtida pelo modelo SIR ajustado aos
# mesmos
# ##############################################################################
plt.scatter(t, data, marker='.', color='yellow', label ='Dados oficiais do MS até 23/06')
plt.plot(solver(np.arange(200), *popt), label='Curva ajustada (modelo SIR) - 23/06 (t = 118)', linewidth=1.0)
plt.text(150, 100000, 'gamma=%1.1f, alpha=%1.2f\ns_0=%d, i_0=%d, a_0=%d' % tuple(popt))
plt.title('Dados oficiais do MS e curvas do modelo SIR ajustadas aos mesmos')
plt.xlabel('Tempo transcorrido desde a primeira infecção [em dias]')
plt.ylabel('Total acumulado de pessoas infectadas')
plt.legend()
plt.show()

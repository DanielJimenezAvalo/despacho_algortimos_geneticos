#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:14:11 2021

@author: daniel
"""
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
import random

dict_potencia_unidades = {}
dict_potencia_unidades[(0,'potencia')]=[0,32,50,87,94]
dict_potencia_unidades[(1,'potencia')]=[0,21,34,100,122]
dict_potencia_unidades[(2,'potencia')]=[0,15,26,54,108]
dict_potencia_unidades[(3,'potencia')]=[0,22,45,55,127]
dict_potencia_unidades[(4,'potencia')]=[0,19,47,58,97]
dict_potencia_unidades[(5,'potencia')]=[0,19,33,72,147]
dict_potencia_unidades[(6,'potencia')]=[0,20,42,74,104]
dict_potencia_unidades[(7,'potencia')]=[0,20,49,71,130]
dict_potencia_unidades[(8,'potencia')]=[0,15,33,62,146]
dict_potencia_unidades[(9,'potencia')]=[0,24,36,60,100]
dict_potencia_unidades[(10,'potencia')]=[0,19,48,80,133]
dict_potencia_unidades[(11,'potencia')]=[0,23,49,68,108]
dict_potencia_unidades[(12,'potencia')]=[0,16,50,78,146]
dict_potencia_unidades[(13,'potencia')]=[0,25,55,81,138]
dict_potencia_unidades[(14,'potencia')]=[0,22,32,80,99]

dict_consumo_unidades = {}
dict_consumo_unidades[(0,'cons_combust')]=[15,62,95,136,175]
dict_consumo_unidades[(1,'cons_combust')]=[15,61,88,140,178]
dict_consumo_unidades[(2,'cons_combust')]=[17,69,111,141,152]
dict_consumo_unidades[(3,'cons_combust')]=[23,60,118,143,191]
dict_consumo_unidades[(4,'cons_combust')]=[19,41,119,142,172]
dict_consumo_unidades[(5,'cons_combust')]=[15,47,81,129,167]
dict_consumo_unidades[(6,'cons_combust')]=[25,46,102,134,186]
dict_consumo_unidades[(7,'cons_combust')]=[24,49,115,141,169]
dict_consumo_unidades[(8,'cons_combust')]=[18,52,84,138,178]
dict_consumo_unidades[(9,'cons_combust')]=[15,67,94,143,182]
dict_consumo_unidades[(10,'cons_combust')]=[20,46,111,141,197]
dict_consumo_unidades[(11,'cons_combust')]=[24,51,80,149,185]
dict_consumo_unidades[(12,'cons_combust')]=[16,57,106,128,158]
dict_consumo_unidades[(13,'cons_combust')]=[16,68,117,142,168]
dict_consumo_unidades[(14,'cons_combust')]=[24,47,96,145,169]

dict_precio_combustibles = {}
dict_precio_combustibles[(0,'precio_comb')]=0.084
dict_precio_combustibles[(1,'precio_comb')]=0.268
dict_precio_combustibles[(2,'precio_comb')]=0.265
dict_precio_combustibles[(3,'precio_comb')]=0.756
dict_precio_combustibles[(4,'precio_comb')]=0.12
dict_precio_combustibles[(5,'precio_comb')]=0.043
dict_precio_combustibles[(6,'precio_comb')]=0.124
dict_precio_combustibles[(7,'precio_comb')]=0.436
dict_precio_combustibles[(8,'precio_comb')]=0.18
dict_precio_combustibles[(9,'precio_comb')]=0.775
dict_precio_combustibles[(10,'precio_comb')]=0.395
dict_precio_combustibles[(11,'precio_comb')]=0.059
dict_precio_combustibles[(12,'precio_comb')]=0.003
dict_precio_combustibles[(13,'precio_comb')]=0.744
dict_precio_combustibles[(14,'precio_comb')]=0.075

DEMANDA = 1200

NUMERO_INDIVIDUOS_SELECCIONADOS = 20
    
def generar_curva_consumo(vector_potencia,vector_consumo,splits):
           
    x = np.linspace(min(vector_potencia), max(vector_potencia),num=splits)
    
    ysp = InterpolatedUnivariateSpline(vector_potencia,vector_consumo)(x)
    
    curva_consumo = list(zip(x,ysp))
    
    return curva_consumo

#dict_curva_consumo={}

#for i in range(len(dict_potencia_unidades)):
    
#    dict_curva_consumo[i,'consumo_combustible'] = generar_curva_consumo(dict_potencia_unidades[(i,'potencia')],
#                                                  dict_consumo_unidades[(i,'cons_combust')],
#                                                  10000)
    
#    dict_curva_consumo[i,'potencia'] = [dict_curva_consumo[(i,'consumo_combustible')][k][0] for k in range(len(dict_curva_consumo[(i,'consumo_combustible')]))] 
    
#    dict_curva_consumo[i,'consumo'] = [dict_curva_consumo[(i,'consumo_combustible')][k][1] for k in range(len(dict_curva_consumo[(i,'consumo_combustible')]))] 

#    dict_curva_consumo[i,'gradiente'] = [((dict_curva_consumo[i,'consumo'][k+1]-dict_curva_consumo[i,'consumo'][k])*dict_precio_combustibles[(i,'precio_comb')])/(dict_curva_consumo[i,'potencia'][k+1]-dict_curva_consumo[i,'potencia'][k]) for k in range(len(dict_curva_consumo[(i,'consumo_combustible')])-1)]
        
#    dict_curva_consumo[i,'gradiente'] = np.hstack((dict_curva_consumo[i,'gradiente'],[1000]))
    
def generar_dict_curva_consumo_parcial(dict_potencia_unidades,dict_consumo_unidades,numero_unidades):
    
    dict_curva_consumo={}

    for i in range(numero_unidades):
        
        dict_curva_consumo[i,'consumo_combustible'] = generar_curva_consumo(dict_potencia_unidades[(i,'potencia')],
                                                      dict_consumo_unidades[(i,'cons_combust')],
                                                      10000)
        
        dict_curva_consumo[i,'potencia'] = [dict_curva_consumo[(i,'consumo_combustible')][k][0] for k in range(len(dict_curva_consumo[(i,'consumo_combustible')]))] 
        
        dict_curva_consumo[i,'consumo'] = [dict_curva_consumo[(i,'consumo_combustible')][k][1] for k in range(len(dict_curva_consumo[(i,'consumo_combustible')]))] 
    
        dict_curva_consumo[i,'gradiente_pos'] = [((dict_curva_consumo[i,'consumo'][k+1]-dict_curva_consumo[i,'consumo'][k])*dict_precio_combustibles[(i,'precio_comb')])/(dict_curva_consumo[i,'potencia'][k+1]-dict_curva_consumo[i,'potencia'][k]) for k in range(len(dict_curva_consumo[(i,'consumo_combustible')])-1)]
            
        dict_curva_consumo[i,'gradiente_pos'] = np.hstack((dict_curva_consumo[i,'gradiente_pos'],[1000]))

        dict_curva_consumo[i,'gradiente_neg'] = [(-(dict_curva_consumo[i,'consumo'][k]-dict_curva_consumo[i,'consumo'][k-1])*dict_precio_combustibles[(i,'precio_comb')])/(dict_curva_consumo[i,'potencia'][k]-dict_curva_consumo[i,'potencia'][k-1]) for k in range(1,len(dict_curva_consumo[(i,'consumo_combustible')]))]
            
        dict_curva_consumo[i,'gradiente_neg'] = np.hstack(([0],dict_curva_consumo[i,'gradiente_neg']))
        
    return dict_curva_consumo
    
dict_curva_consumo = generar_dict_curva_consumo_parcial(dict_potencia_unidades,dict_consumo_unidades,numero_unidades=15)
#plt.plot(dict_curva_consumo[8,'potencia'],dict_curva_consumo[8,'consumo'])
#plt.show()

def generar_cromosomas_valor(valor,vector):
    
    if valor != None:
        
        posicion = np.where(vector == valor)[0][0]
        
    else:
        
        posicion = random.randint(0,np.shape(vector)[0]-1)
        
    cromosoma = np.zeros(np.shape(vector)[0])
    
    cromosoma[posicion] = 1

    return cromosoma
    
def generar_dict_inicial_cromosomas(dict_curva_consumo,dict_potencia_unidades):
    
    dict_inicial_cromosomas={}
    
    for i in range(len(dict_potencia_unidades)):
        
        dict_inicial_cromosomas[(i,'cromosoma')]=generar_cromosomas_valor(None,dict_curva_consumo[i,'potencia'])
        
    return dict_inicial_cromosomas

dict_cromosomas_inicial = generar_dict_inicial_cromosomas(dict_curva_consumo,dict_potencia_unidades)

#dict_despacho_generado={}

#potencia_despachada = 0

#costo_despachado = 0

#for i in range(len(dict_potencia_unidades)):
    
#    dict_despacho_generado[(i,'potencia_generada')] = np.dot(dict_cromosomas_inicial[(i,'cromosoma')],dict_curva_consumo[(i,'potencia')])
    
#    dict_despacho_generado[(i,'gradiente_generada')] = np.dot(dict_cromosomas_inicial[(i,'cromosoma')],dict_curva_consumo[(i,'gradiente')])
    
#    dict_despacho_generado[(i,'potencia_disponible')] = max(dict_potencia_unidades[(i,'potencia')])-dict_despacho_generado[(i,'potencia_generada')]
    
#    dict_despacho_generado[(i,'consumo')] = np.dot(dict_cromosomas_inicial[(i,'cromosoma')],dict_curva_consumo[(i,'consumo')])
    
#    dict_despacho_generado[(i,'costo')] = dict_despacho_generado[(i,'consumo')]*dict_precio_combustibles[(i,'precio_comb')]
    
#    potencia_despachada += dict_despacho_generado[(i,'potencia_generada')]
    
#    costo_despachado += dict_despacho_generado[(i,'costo')]
    
def generar_despacho(dict_cromosomas,dict_curva_consumo):
    
    dict_despacho_generado={}

    potencia_despachada = 0
    
    costo_despachado = 0
    
    for i in range(len(dict_potencia_unidades)):
        
        dict_despacho_generado[(i,'potencia_generada')] = np.dot(dict_cromosomas_inicial[(i,'cromosoma')],dict_curva_consumo[(i,'potencia')])
        
        dict_despacho_generado[(i,'gradiente_generada_pos')] = np.dot(dict_cromosomas_inicial[(i,'cromosoma')],dict_curva_consumo[(i,'gradiente_pos')])

        dict_despacho_generado[(i,'gradiente_generada_neg')] = np.dot(dict_cromosomas_inicial[(i,'cromosoma')],dict_curva_consumo[(i,'gradiente_neg')])
        
        dict_despacho_generado[(i,'potencia_disponible')] = max(dict_potencia_unidades[(i,'potencia')])-dict_despacho_generado[(i,'potencia_generada')]
        
        dict_despacho_generado[(i,'consumo')] = np.dot(dict_cromosomas_inicial[(i,'cromosoma')],dict_curva_consumo[(i,'consumo')])
        
        dict_despacho_generado[(i,'costo')] = dict_despacho_generado[(i,'consumo')]*dict_precio_combustibles[(i,'precio_comb')]
        
        potencia_despachada += dict_despacho_generado[(i,'potencia_generada')]
    
        costo_despachado += dict_despacho_generado[(i,'costo')]  
        
    return potencia_despachada, costo_despachado, dict_despacho_generado

potencia_despachada, costo_despachado, dict_despacho_generado = generar_despacho(dict_cromosomas_inicial,dict_curva_consumo)



capacidad_sistema=0

for i in range(len(dict_potencia_unidades)):
    
    capacidad_sistema += max(dict_potencia_unidades[(i,'potencia')])
    
if DEMANDA <= capacidad_sistema:
    
    print('Ejecutar despacho')
    
else:
    
    print('racionamiento')

factible = 0
    
while factible<0:
    
    dict_curva_consumo = generar_dict_curva_consumo_parcial(dict_potencia_unidades,dict_consumo_unidades,numero_unidades=15)

    dict_cromosomas_inicial = generar_dict_inicial_cromosomas(dict_curva_consumo,dict_potencia_unidades)

    potencia_despachada, costo_despachado, dict_despacho_generado = generar_despacho(dict_cromosomas_inicial,dict_curva_consumo)

    if potencia_despachada<=1.005*DEMANDA and potencia_despachada>=0.99*DEMANDA:
        
        factible += 1
    
    elif potencia_despachada>=1.005*DEMANDA:
        
        factible += 1
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    

    
    

    


























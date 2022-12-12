#from typing import TYPE_CHECKING
import numpy as np
import random
import pandas as pd

p1=np.array([0,10,20,30,40,50])
c1=np.array([0,100,200,300,400,500])
curva_c1=list(zip(p1,c1))

p2=np.array([0,10,20,30])
c2=np.array([0,50,80,100])
curva_c2=list(zip(p2,c2))

p3=np.array([0,10,20,60,80,100])
c3=np.array([0,50,100,200,600,800])
curva_c3=list(zip(p3,c3))


#p3_val=80
#pos_p3_val=np.where(p3==p3_val)[0][0]
#c3_val=c3[pos_p3_val]


def valor_curva_costo(px_val,px,cx):
    pos_px_val=np.where(px==px_val)[0][0]
    cx_val=cx[pos_px_val]
    return cx_val

#c3_val=valor_curva_costo(p3_val,p3,c3)

def gen_cromosoma_pos(px_val,px):
    pos_px_val=np.where(px==px_val)[0][0]
    px_cromosoma=np.zeros(np.shape(px)[0])
    px_cromosoma[pos_px_val]=1
    return px_cromosoma

    
D=100

#factibilidad
'''
p1_long=np.shape(p1)[0]
p2_long=np.shape(p2)[0]
p3_long=np.shape(p3)[0]

pos_p1_int=random.randint(0,p1_long-1)
pos_p2_int=random.randint(0,p2_long-1)
pos_p3_int=random.randint(0,p3_long-1)

p1_cromosoma = np.zeros(p1_long)
p2_cromosoma = np.zeros(p2_long)
p3_cromosoma = np.zeros(p3_long)

p1_cromosoma[pos_p1_int]=1
p2_cromosoma[pos_p2_int]=1
p3_cromosoma[pos_p3_int]=1
'''

def generar_cromosomas(px):
    px_long=np.shape(px)[0]
    pos_px_int=random.randint(0,px_long-1)
    px_cromosoma=np.zeros(px_long)
    px_cromosoma[pos_px_int]=1
    return px_cromosoma

#p1_long=np.shape(p1)[0] 
#p1_cromosoma=generar_cromosomas(p1)
#p2_cromosoma=generar_cromosomas(p2)
#p3_cromosoma=generar_cromosomas(p3)

'''
p1_val=np.multiply(p1,p1_cromosoma)
p2_val=np.multiply(p2,p2_cromosoma)
p3_val=np.multiply(p3,p3_cromosoma)

c1_val=np.multiply(c1,p1_cromosoma)
c2_val=np.multiply(c2,p2_cromosoma)
c3_val=np.multiply(c3,p3_cromosoma)
'''

def generar_pot_y_cost(px,cx,px_cromosoma):
    px_val=np.dot(px,px_cromosoma)
    cx_val=np.dot(cx,px_cromosoma)
    return px_val, cx_val

#(p1_val,c1_val)=generar_pot_y_cost(p1,c1,p1_cromosoma)
#(p2_val,c2_val)=generar_pot_y_cost(p2,c2,p2_cromosoma)
#(p3_val,c3_val)=generar_pot_y_cost(p3,c3,p3_cromosoma)

#encontrar todos los teminos factibles
factibles=0
poblacion=0
dict_factibles={}
dict_seleccion_costo={}
dict_seleccion_potencia={}

while factibles<30:
    
    p1_cromosoma=generar_cromosomas(p1)
    p2_cromosoma=generar_cromosomas(p2)
    
    (p1_val,c1_val)=generar_pot_y_cost(p1,c1,p1_cromosoma)
    (p2_val,c2_val)=generar_pot_y_cost(p2,c2,p2_cromosoma)
    
    p3_val = D - p1_val -p2_val
        
    if p3_val in p3:
        
        c3_val=valor_curva_costo(p3_val,p3,c3)
        
        p3_cromosoma=gen_cromosoma_pos(p3_val,p3)
              
        dict_factibles[(factibles,'p1_cromosomas')]=p1_cromosoma
        dict_factibles[(factibles,'p2_cromosomas')]=p2_cromosoma
        dict_factibles[(factibles,'p3_cromosomas')]=p3_cromosoma
        
        dict_factibles[(factibles,'potencia')]=[p1_val,p2_val,p3_val]
        dict_factibles[(factibles,'costo')]=[c1_val,c2_val,c3_val]
        dict_factibles[(factibles,'costo_total')]=np.sum(dict_factibles[(factibles,'costo')])
        
        dict_seleccion_costo[factibles]=dict_factibles[(factibles,'costo_total')]
        dict_seleccion_potencia[factibles]=np.hstack((p1_cromosoma,p2_cromosoma,p3_cromosoma))
        
        factibles+=1
    

def sort_dict_by_value(dict_selec):
    sort_dict_select=sorted(dict_selec.items(), key=lambda x:x[1])
    return sort_dict_select

dict_sort_sel_costo=sort_dict_by_value(dict_seleccion_costo)

### hacer crossover 

parent1_factible=dict_sort_sel_costo[0][0]

parent2_factible=dict_sort_sel_costo[5][0]

cromosoma_padre=dict_seleccion_potencia[parent1_factible]

cromosoma_madre=dict_seleccion_potencia[parent2_factible]

#cromosoma_p_1=cromosoma_padre[0:np.shape(p1)[0]]   
    
#cromosoma_p_2=cromosoma_padre[np.shape(p1)[0]:np.shape(p1)[0]+np.shape(p2)[0]]  

#cromosoma_p_3=cromosoma_padre[np.shape(p1)[0]+np.shape(p2)[0]:np.shape(p1)[0]+np.shape(p2)[0]+np.shape(p3)[0]]  

def hacer_crossover(cromosoma_parent_1,cromosoma_parent2):
    
    cromosoma_hijo_1=np.hstack((cromosoma_madre[0:np.shape(p1)[0]], cromosoma_padre[np.shape(p1)[0]:np.shape(p1)[0]+np.shape(p2)[0]+np.shape(p3)[0]]))

    cromosoma_hijo_2=np.hstack((cromosoma_padre[0:np.shape(p1)[0]], cromosoma_madre[np.shape(p1)[0]:np.shape(p1)[0]+np.shape(p2)[0]+np.shape(p3)[0]]))
  
    return cromosoma_hijo_1 , cromosoma_hijo_2


cromosoma_hijo_1=np.hstack((cromosoma_madre[0:np.shape(p1)[0]], cromosoma_padre[np.shape(p1)[0]:np.shape(p1)[0]+np.shape(p2)[0]+np.shape(p3)[0]]))

cromosoma_hijo_2=np.hstack((cromosoma_padre[0:np.shape(p1)[0]], cromosoma_madre[np.shape(p1)[0]:np.shape(p1)[0]+np.shape(p2)[0]+np.shape(p3)[0]]))

########################################

hijo1_p1_cromosoma = cromosoma_hijo_1[0:np.shape(p1)[0]]

hijo1_p2_cromosoma = cromosoma_hijo_1[np.shape(p1)[0]:np.shape(p1)[0]+np.shape(p2)[0]]

hijo1_p3_cromosoma = cromosoma_hijo_1[np.shape(p1)[0]+np.shape(p2)[0]:np.shape(p1)[0]+np.shape(p2)[0]+np.shape(p3)[0]]


(hijo1_p1_val,hijo1_c1_val) = generar_pot_y_cost(p1,c1,hijo1_p1_cromosoma)

(hijo1_p2_val,hijo1_c2_val) = generar_pot_y_cost(p2,c2,hijo1_p2_cromosoma)

(hijo1_p3_val,hijo1_c3_val) = generar_pot_y_cost(p3,c3,hijo1_p3_cromosoma)


### hacer mutacion

if D != hijo1_p1_val+hijo1_p2_val+hijo1_p3_val:
    
    mutado_hijo1_p1_val = D-(hijo1_p2_val+hijo1_p3_val)
    
    mutado_hijo1_p2_val = D-(hijo1_p1_val+hijo1_p3_val)
    
    mutado_hijo1_p3_val = D-(hijo1_p1_val+hijo1_p2_val)
    
    if mutado_hijo1_p3_val in p3:
        
        mutado_hijo1_c3_val = valor_curva_costo(mutado_hijo1_p3_val,p3,c3)

        p1_mutado_cromosoma = gen_cromosoma_pos(hijo1_p1_val,p1)
        
        p2_mutado_cromosoma = gen_cromosoma_pos(hijo1_p2_val,p2)
        
        p3_mutado_cromosoma = gen_cromosoma_pos(mutado_hijo1_p3_val,p3)
            
#        (p1_mutado_val,c1_mutado_val) = generar_pot_y_cost(p1,c1,p1_mutado_cromosoma)
        
#        (p2_mutado_val,c2_mutado_val) = generar_pot_y_cost(p2,c2,p2_mutado_cromosoma)
        
#        (p3_mutado_val,c3_mutado_val) = generar_pot_y_cost(p2,c2,p3_mutado_cromosoma)
      
    elif mutado_hijo1_p2_val in p2:
        
        mutado_hijo1_c2_val = valor_curva_costo(mutado_hijo1_p2_val,p2,c2)

        p1_mutado_cromosoma = gen_cromosoma_pos(hijo1_p1_val,p1)
        
        p2_mutado_cromosoma = gen_cromosoma_pos(mutado_hijo1_p2_val,p2)
        
        p3_mutado_cromosoma = gen_cromosoma_pos(hijo1_p3_val,p3)
    
#        (p1_mutado_val,c1_mutado_val) = generar_pot_y_cost(p1,c1,p1_mutado_cromosoma)
        
#        (p2_mutado_val,c2_mutado_val) = generar_pot_y_cost(p2,c2,p2_mutado_cromosoma)
        
#        (p3_mutado_val,c3_mutado_val) = generar_pot_y_cost(p2,c2,p3_mutado_cromosoma)
    
    elif mutado_hijo1_p1_val in p1:

        mutado_hijo1_c1_val = valor_curva_costo(mutado_hijo1_p1_val,p1,c1)

        p1_mutado_cromosoma = gen_cromosoma_pos(mutado_hijo1_p1_val,p1)
        
        p2_mutado_cromosoma = gen_cromosoma_pos(hijo1_p2_val,p2)
        
        p3_mutado_cromosoma = gen_cromosoma_pos(hijo1_p3_val,p3)
    
#        (p1_mutado_val,c1_mutado_val) = generar_pot_y_cost(p1,c1,p1_mutado_cromosoma)
        
#        (p2_mutado_val,c2_mutado_val) = generar_pot_y_cost(p2,c2,p2_mutado_cromosoma)
        
#        (p3_mutado_val,c3_mutado_val) = generar_pot_y_cost(p2,c2,p3_mutado_cromosoma)
    
    else:
        0
        
def do_mutating(cromosoma,demanda,p1_val,p2_val,p3_val,p1,p2,p3):
    
    if demanda != p1_val + p2_val + p3_val:
        
        p1_val_mutado = D - (p2_val + p3_val)
        
        p2_val_mutado = D - (p1_val + p3_val)
        
        p3_val_mutado = D - (p1_val + p1_val)
        
        if p1_val_mutado in p1:
            
            p1_cromosoma_mutado_1 = gen_cromosoma_pos(p1_val_mutado,p1)
            
            p2_cromosoma_mutado_1 = gen_cromosoma_pos(p2_val,p2)
            
            p3_cromosoma_mutado_1 = gen_cromosoma_pos(p3_val,p3)
            
            cromosoma_mutado_1 = np.hstack((p1_cromosoma_mutado_1,
                                            p2_cromosoma_mutado_1,
                                            p3_cromosoma_mutado_1))
            
        elif p2_val_mutado in p2:
            
            p1_cromosoma_mutado_2 = gen_cromosoma_pos(p1_val,p1)
            
            p2_cromosoma_mutado_2 = gen_cromosoma_pos(p2_val_mutado,p2)
            
            p3_cromosoma_mutado_2 = gen_cromosoma_pos(p3_val,p3)
            
            cromosoma_mutado_2 = np.hstack((p1_cromosoma_mutado_2,
                                            p2_cromosoma_mutado_2,
                                            p3_cromosoma_mutado_2))
            
        elif p3_val_mutado in p3:
            
            p1_cromosoma_mutado_3 = gen_cromosoma_pos(p1_val,p1)
            
            p2_cromosoma_mutado_3 = gen_cromosoma_pos(p2_val,p2)
            
            p3_cromosoma_mutado_3 = gen_cromosoma_pos(p3_val_mutado,p3)
            
            cromosoma_mutado_3 = np.hstack((p1_cromosoma_mutado_3,
                                            p2_cromosoma_mutado_3,
                                            p3_cromosoma_mutado_3))
        
        else:
            0
    
    
def fitness(cromosoma,)       
    
    

        
    
     
    
        
    
    

    
    
    
    
    
    



print(dict_factibles)

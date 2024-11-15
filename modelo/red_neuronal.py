import random as rnd
import math
import numpy as np

class red:
    def __init__(self,ncE,ncO,ncS,s="sigma",r=0.7):
        self.capaEntrada=[]
        self.capaOculta=[]
        self.capaSalida=[]
        for i in range(ncE):
            self.capaEntrada.append(neurona(1,s,r))
        for i in range(ncO):
            self.capaOculta.append(neurona(ncE,s,r))
        for i in range(ncS):
            self.capaSalida.append(neurona(ncO,s,r))
       
        self.salidaEntrada=[]
        self.salidaOculta=[]
        self.Y=[]
        
       
       
       
    def calculaSalida(self,X):
        self.salidaEntrada=[]
        self.salidaOculta=[]
        self.Y=[]
        #Procesar capa entrada
        for i in range(len(self.capaEntrada)):
            self.salidaEntrada.append(self.capaEntrada[i].calculaSalida([X[i]]))
       
        #Procesar capa entrada
        for i in range(len(self.capaOculta)):
            self.salidaOculta.append(self.capaOculta[i].calculaSalida(self.salidaEntrada))
       
        #Procesar capa salida
        for i in range(len(self.capaSalida)):
            self.Y.append(self.capaSalida[i].calculaSalida(self.salidaOculta))
        return self.Y
           
    def entrena(self,X,T):
       
        self.calculaSalida(X)
       
        #Cálculo corrección Salida
       
        #b1 1)
        deltaSalida = []
        for i in range(len(self.Y)):
            deltaSalida.append((T[i]-self.Y[i])*self.Y[i]*(1-self.Y[i]))
       
        #b1 2)
        deltaWS=[]
        for x in self.salidaOculta:
            fila = []
            for j in range(len(self.capaSalida)):    
                fila.append(self.capaSalida[j].r*deltaSalida[j]*x)
            deltaWS.append(fila)
       
        #Cálculos corrección Oculta y Entrada
        #b2 1)
        deltaOculta=[]
        for i in range(len(self.capaOculta)):
            suma=0
            for j in range(len(self.capaSalida)):
                suma+=deltaSalida[j]*self.capaSalida[j].w[i]
            deltaOculta.append(self.capaOculta[i].y*(1-self.capaOculta[i].y)*suma)
       
        #b2 2)
        deltaWO=[]
        for x in self.salidaEntrada:
            fila=[]
            for i in range(len(self.capaOculta)):
                fila.append(self.capaOculta[i].r*deltaOculta[i]*x)
            deltaWO.append(fila)
       
        #b2 3)
        deltaEntrada=[]
        for k in range(len(self.capaEntrada)):
            suma=0
            for i in range(len(self.capaOculta)):
                suma+=deltaOculta[i]*self.capaOculta[i].w[k]
            deltaEntrada.append(self.capaEntrada[k].y*(1-self.capaEntrada[k].y)*suma)
       
        #b2 4)
        deltaWE=[]
        for k in range(len(self.capaEntrada)):
            deltaWE.append(self.capaEntrada[k].r*deltaEntrada[k]*X[k])
       
        #Ajuste de pesos
        #b3 1)
        for j in range (len(self.capaSalida)):      
            for i in range(len(self.capaOculta)):
                self.capaSalida[j].w[i] += deltaWS[i][j]        
        #b3 2)
        for i in range(len(self.capaOculta)):
            for k in range(len(self.capaEntrada)):
                self.capaOculta[i].w[k]+=deltaWO[k][i]
        #b3 3)
        for k in range(len(self.capaEntrada)):
            self.capaEntrada[k].w[0]+=deltaWE[k]
       
        #Ajuste Sesgo
        #(11)
        for j in range (len(self.capaSalida)):
            self.capaSalida[j].theta += self.capaSalida[j].r*deltaSalida[j]
        #(12)
        for i in range(len(self.capaOculta)):
            self.capaOculta[i].theta+=self.capaOculta[i].r*deltaOculta[i]
        #(13)
        for k in range(len(self.capaEntrada)):
            self.capaEntrada[k].theta+=self.capaEntrada[k].r*deltaEntrada[k]


       
class neurona:
    def __init__(self,nI,s="sigma",r=0.7):
        self.r=r
        self.nI=nI
        self.w=[]
        for i in range(nI):
            self.w.append(rnd.random())
        self.x=[]
        for i in range(nI):
            self.x.append(0)
        self.theta=rnd.random()
        self.s=s
        self.y=0
   
    def fA(self,x):
        if(self.s=="escalon"):
            if(x>0):
                return 1
            else:
                return 0
        if(self.s=="rampa"):
            if(x<0):
                return 0
            elif(x<1):
                return x
            else:
                return 1
        if(self.s=="sigma"):
            if(x<-10000):
                return 0
            elif(x>10000):
                return 1
            else:
                return 1/(1+1/math.exp(x))
           
   
    def calculaSalida(self,X):
        self.x=X
        suma=0
        for i in range(self.nI):
            suma+=self.w[i]*self.x[i]
        suma+=self.theta
        self.y=self.fA(suma)
        return self.y
   
    def entrena(self,X,t):
        self.calculaSalida(X)
       
        #b1 1)
        deltaSalida=(t-self.y)*self.y*(1-self.y)
       
        #b1 2)
        deltaW=[]
        for x in X:
            deltaW.append(self.r*deltaSalida*x)
       
        #b3 1)
        for i in range(len(self.w)):
            self.w[i]+=deltaW[i]
       
        #(11)
        self.theta+=self.r*deltaSalida
       

    def ejemplo():
        entradas=[[0,0,0],
                [0,0,1],
                [0,0,0],
                [1,0,0],
                [1,0,1],
                [1,1,1],
                [0,0,0],
                [0,0,0],
                [1,1,0]]

        salidas=[0,1,0,1,0,0,0,0,0]

        r = red(3,4,1)

        for i in range(1000):
            for j in range(len(entradas)):
                r.entrena(entradas[j], [salidas[j]])


        #n=neurona(3,r=0.3)

        # for i in range(100000):
        #     for j in range(len(entradas)):
        #         n.entrena(entradas[j], salidas[j])
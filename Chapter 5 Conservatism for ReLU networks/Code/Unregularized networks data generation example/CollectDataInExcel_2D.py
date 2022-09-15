# Running file to train multiple networks and save the result in Excel
import os


for i in range(1):
    exec(open("./Train2LayerNetworks2DToyExample.py").read())
    exec(open("./LipBaB_2D.py").read())
    exec(open("./LipBab_forActivationPatterns_2D.py").read())

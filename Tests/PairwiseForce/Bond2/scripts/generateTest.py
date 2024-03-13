import sys,os

import pyUAMMD

import math
import numpy as np
import json
import jsbeautifier

#Import the json input file
with open("parameters.json", "r") as f:
    param = json.load(f)

#Read the parameters
N              = 2
temperature    = 0

pos0 = param["pos0"]
pos1 = param["pos1"]
L    = param["L"]
k    = param["k"]
r0   = param["r0"]
box         = [L,L,L]
temperature = 0

#Create simulation

simulation = pyUAMMD.simulation()
simulation["system"] = {}
simulation["system"]["info"] = {}
simulation["system"]["info"]["type"] = ["Simulation","Information"]
simulation["system"]["info"]["parameters"] = {}
simulation["system"]["info"]["parameters"]["name"] = "testHessian"

simulation["global"] = {}
simulation["global"]["types"] = {}
simulation["global"]["types"]["type"]   = ["Types","Basic"]
simulation["global"]["types"]["labels"] = ["name", "radius", "mass", "charge"]
simulation["global"]["types"]["data"]  = [["A", 1, 0, 0]]

simulation["global"]["ensemble"] = {}
simulation["global"]["ensemble"]["type"]   = ["Ensemble","NVT"]
simulation["global"]["ensemble"]["labels"] = ["box", "temperature"]
simulation["global"]["ensemble"]["data"]   = [[box, temperature]]

simulation["integrator"] = {}
simulation["integrator"]["BBK"] = {}
simulation["integrator"]["BBK"]["type"] = ["Langevin", "BBK"]
simulation["integrator"]["BBK"]["parameters"] = {}
simulation["integrator"]["BBK"]["parameters"]["timeStep"] = 0.001
simulation["integrator"]["BBK"]["parameters"]["frictionConstant"] = 1

simulation["integrator"]["schedule"] = {}
simulation["integrator"]["schedule"]["type"] = ["Schedule", "Integrator"]
simulation["integrator"]["schedule"]["labels"] = ["order", "integrator","steps"]
simulation["integrator"]["schedule"]["data"] = [
    [1, "BBK", 1],
]

simulation["state"] = {}
simulation["state"]["labels"] = ["id", "position"]
simulation["state"]["data"] = [[0, pos0],
                               [1, pos1]]


simulation["topology"] = {}
simulation["topology"]["forceField"] = {}
simulation["topology"]["forceField"]["Bond"] = {}
simulation["topology"]["forceField"]["Bond"]["labels"] = ["id_i", "id_j", "K", "r0"]
simulation["topology"]["forceField"]["Bond"]["data"] = [[0, 1, k, r0]]

simulation["topology"]["forceField"]["Bond"]["parameters"] = {}
simulation["topology"]["forceField"]["Bond"]["type"] = ["Bond2", "Harmonic"]

simulation["topology"]["structure"] = {}
simulation["topology"]["structure"]["labels"] = ["id", "type"]
simulation["topology"]["structure"]["data"] = [[0, "A"],
                                               [1, "A"]]



# #Output

simulation["simulationStep"] = {}

name = "Bond2PairForces"
simulation["simulationStep"][name] = {}
simulation["simulationStep"][name]["type"] = ["MechanicalMeasure", "PairwiseForceMeasure"]
simulation["simulationStep"][name]["parameters"] = {}
simulation["simulationStep"][name]["parameters"]["outputFilePath"]     = "PairForces.out"
simulation["simulationStep"][name]["parameters"]["intervalStep"]       = 1
simulation["simulationStep"][name]["parameters"]["mode"]       = "Pairwise_force"
                                                                    
name = "BondTotalForces"
simulation["simulationStep"][name] = {}
simulation["simulationStep"][name]["type"] = ["MechanicalMeasure", "PairwiseForceMeasure"]
simulation["simulationStep"][name]["parameters"] = {}
simulation["simulationStep"][name]["parameters"]["outputFilePath"]     = "TotalForces.out"
simulation["simulationStep"][name]["parameters"]["mode"]               = "Total_force"
simulation["simulationStep"][name]["parameters"]["intervalStep"]       = 1

#Check if ./results folder exists, if not create it
if not os.path.exists("./results"):
    os.makedirs("./results")
simulation.write("./results/test.json")

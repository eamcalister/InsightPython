# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:09:03 2024

@author: mcalister_e
"""

#This is an initial attempt at writing a Python script to take a model built in Insightmaker and get it into Python for more advanced analysis. 
#This one is attempting to store stocks, flows, and variables as data frames.

import numpy as np
import pandas as pd
from sympy import symbols,sympify,lambdify
import scipy.integrate
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
#from utils.ode_tools import plot_phase_sol

#Mapping between Insightmaker functions and sympy versions.
function_mapping = {
    #Functions with single inputs.
    'Sin': 'sin',
    'Cos': 'cos',
    'Tan': 'tan',
    'ArcSin': 'asin',
    'ArcCos': 'acos',
    'ArcTan': 'atan',
    'Exp': 'exp',
    'Ln':'log',
    'Abs':'Abs',
    'Floor':'floor',
    'Ceiling':'ceiling',
    'Round':'RoundFunction',
    'Sqrt':'sqrt',
        
    #times are all just t.
    'Seconds()': 't',
    'Minutes()': 't',
    'Hours()': 't',
    'Days()':'t',
    'Months()':'t',
    'Years()':'t'
    # Add more mappings as needed
}

#Actual Insightmaker to Sympy conversion function.
def convert_expression(expression):
    for insight_maker_function, sympy_function in function_mapping.items():
        expression = expression.replace(insight_maker_function, sympy_function)
    return expression

#To do: Do a second set for various functions that take arguments in specific ways, e.g. Step to Heaviside.


#Function to build stocks data frame,

def parse_insightmaker_stocks(model_path):
    with open(model_path, 'r') as file:
        model_content = file.read()

    soup = BeautifulSoup(model_content, 'xml')
    
    stock_cols = ['name','initial_value','identity']
    stocks = pd.DataFrame(columns = stock_cols)

#Not available attributes right now: Units, conveyor (0 or 1), non-negativity, etc. We are creating a "naked" system of ODES 
    for variable in soup.find_all('Stock'):
        name1 = variable.get('name')
        name = name1.replace(' ','_')
        initial_value = variable.get('InitialValue')
        identity = variable.get('id')
        new_row = [name,initial_value,identity]
        stocks.loc[len(stocks)] = new_row

    return stocks

#Function to build flows data frame,

def parse_insightmaker_flows(model_path):
    with open(model_path, 'r') as file:
        model_content = file.read()

    soup = BeautifulSoup(model_content, 'xml')
    
    flow_cols = ['name','FlowRate','identity','source','target']
    flows = pd.DataFrame(columns = flow_cols)

#Not available attributes right now: Only positive rates. This should be added later. However, in most cases it is heuristic.
    for variable in soup.find_all('Flow'):
        name = variable.get('name')
        rate0 = variable.get('FlowRate')
        rate1 = rate0.replace('[','').replace(']','').replace(' ','_') #need to create replacement catalog script for functions.
        rate = convert_expression(rate1)
        identity = variable.get('id')
        for cell in variable.find_all('mxCell'): 
            target = cell.get('target')
            source = cell.get('source') #when source or target returns 'None', that's the ether.
        new_row = [name,rate,identity,source,target]
        flows.loc[len(flows)] = new_row
        

    return flows

#Function to build variables data frame,

def parse_insightmaker_variables(model_path):
    with open(model_path, 'r') as file:
        model_content = file.read()

    soup = BeautifulSoup(model_content, 'xml')

    variables_cols= ['name','equation']
    variables = pd.DataFrame(columns = variables_cols)

    for variable in soup.find_all('Variable'):
        name1 = variable.get('name')
        name = name1.replace(' ','_')
        equation0 = variable.get('Equation')
        equation1 = equation0.replace('[','').replace(']','').replace(' ','_')
        if equation1.replace('.', '', 1).isdigit():
            equation = equation1
        else:
            equation = convert_expression(equation1)
        new_row = [name,equation]
        variables.loc[len(variables)] = new_row
        
        
    return variables

#As of right now, we need to just have variables be parameters, which is constraining. To allow for other functions we have to establish a dictionary of Insightmaker to Numpy functions. 

#Function to build settings data frame,

def parse_insightmaker_settings(model_path):
    with open(model_path, 'r') as file:
        model_content = file.read()

    soup = BeautifulSoup(model_content, 'xml')

    settings = np.zeros(3) #settings sets up the t variable for the DEs

    for variable in soup.find_all('Setting'):
        settings[0] = variable.get('TimeStart')
        settings[1] = variable.get('TimeLength')
        settings[2] = variable.get('TimeStep')
        
    return settings

#Now parse the model and create the data frames.
#################################################
#This is where you input the file.
#################################################
if __name__ == "__main__":
    # Replace 'your_insightmaker_model.xml' with the actual path to your Insightmaker model file
    insightmaker_model_path = 'logistic.InsightMaker'
    
    # Parse Insightmaker model
    model_variables = parse_insightmaker_variables(insightmaker_model_path)
    model_stocks = parse_insightmaker_stocks(insightmaker_model_path)
    model_flows = parse_insightmaker_flows(insightmaker_model_path)
    simulation_settings = parse_insightmaker_settings(insightmaker_model_path)

#These three data frames essentially contain  the model.    

# Setup the grid
tspan = np.linspace(simulation_settings[0], simulation_settings[0]+simulation_settings[1], 1+int(np.floor(simulation_settings[1]/simulation_settings[2])))  # np.linspace(initial, end, number_values)

X0 = model_stocks.initial_value #initial values vector

# Filter out only the elements in the variables that are constants. These are the parameters vs. external forces.
model_parameters = model_variables[model_variables['equation'].apply(lambda x: x.replace('.', '', 1).isdigit())]
# Resetting index to have a clean index for the new DataFrame
model_parameters = model_parameters.reset_index(drop=True)

# Filter out only the elements in the variables that are not constants. These are the external forces.
model_externals = model_variables[~model_variables['equation'].apply(lambda x: x.replace('.', '', 1).isdigit())]
# Resetting index to have a clean index for the new DataFrame
model_externals = model_externals.reset_index(drop=True)


#create sympy symbols for all the stocks, parameters, and time. 
stocks = symbols(model_stocks.name.tolist()) 

#Only create a collection of parameters if they exist. 
if len(model_parameters)>0:
    parameter_symbols = symbols(model_parameters.name.tolist()) 
    args = np.array(model_parameters.equation, dtype = float)


t = symbols('t')

#Replace any parameter names that got changed to their expression.
for j in range(len(model_flows.FlowRate)):
    for i in range(len(model_externals)):
        pattern = re.compile(re.escape(model_externals.name[i]))
        model_flows.FlowRate[j] = pattern.sub(model_externals.equation[i], model_flows.FlowRate[j])

#Create the derivatives.
stocks_dot = []
for i in range(len(stocks)):
    d=0
    for j in range(len(model_flows)):
        if model_flows.target[j] == model_stocks.identity[i]:
            d = d+sympify(model_flows.FlowRate[j])
        elif model_flows.source[j] == model_stocks.identity[i]:
            d = d-sympify(model_flows.FlowRate[j])
    stocks_dot.append(d)
 
#Now we need to put these together so scipy can integrate numerically.
SolMethod = 'DOP853'

#Version 1: All parameter values and initial conditions taken from the .Insightmaker file.
if len(model_parameters)>0:
    model = lambdify((t, stocks) + tuple(parameter_symbols), stocks_dot)
    solution = scipy.integrate.solve_ivp(model, (simulation_settings[0], simulation_settings[0]+simulation_settings[1]), X0,method=SolMethod, t_eval = tspan, args = args)
else: 
    model = lambdify((t, stocks), stocks_dot)
    solution = scipy.integrate.solve_ivp(model, (simulation_settings[0], simulation_settings[0]+simulation_settings[1]), X0,method=SolMethod, t_eval = tspan)

#methods: 'RK23', 'RK45', 'DOP853' for non-stiff. 'Radau', 'BDF' for stiff. 'LASODA' switches automatically.



########################################
#Below this you will play with plotting etc.
#This is the point of this entire exercise.
#Notes: 
#1. Import of the .Insightmaker file into Python does not support units of any kind; they must live in Insightmaker.
#2. One time unit should be used throughout your model.
#3. Stocks in your Insightmaker model should have a name and an initial value that is a number; no expressions are allowed in the "Equations" field.

########################################


# Plot the solution
plt.plot(tspan, solution.y[1])
plt.show()



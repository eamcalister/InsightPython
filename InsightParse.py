# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:34:24 2024

@author: mcalister_e
"""

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

#This is a module to call the necessary functions to parse the .Insightmaker file.


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
        name = variable.get('name') #flow names are never really used, just source and targets
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

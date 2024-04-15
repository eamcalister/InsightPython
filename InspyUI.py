import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from sympy import symbols, sympify, lambdify
import scipy.integrate
from bs4 import BeautifulSoup
import re


#This is a module to call the necessary functions to parse the .Insightmaker file.
from InsightParse import *



class ModelExecutorApp:
    def __init__(self, master):
        self.master = master
        master.title("Insightmaker Model Executor")

        self.execute_button = tk.Button(master, text="Execute Model", command=self.execute_model)
        self.execute_button.pack(pady=10)
        
        self.execute_button = tk.Button(master, text="Show ODEs", command=self.ShowODEs)
        self.execute_button.pack(pady=10)

        self.browse_button = tk.Button(master, text="Browse Model", command=self.browse_model)
        self.browse_button.pack(pady=10)

        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack()

        self.model_path = None

    def browse_model(self):
        self.model_path = filedialog.askopenfilename(title="Select Insightmaker Model", filetypes=[("Insightmaker files", "*.InsightMaker")])
        
    def ShowODEs(self):
        if self.model_path:
            # Existing code for parsing and solving the Insightmaker model
            model_variables = parse_insightmaker_variables(self.model_path)
            model_stocks = parse_insightmaker_stocks(self.model_path)
            model_flows = parse_insightmaker_flows(self.model_path)
            simulation_settings = parse_insightmaker_settings(self.model_path)

            tspan = np.linspace(simulation_settings[0], simulation_settings[0] + simulation_settings[1],
                                1 + int(np.floor(simulation_settings[1] / simulation_settings[2])))
            X0 = model_stocks.initial_value

            model_parameters = model_variables[model_variables['equation'].apply(lambda x: x.replace('.', '', 1).isdigit())]
            model_parameters = model_parameters.reset_index(drop=True)

            model_externals = model_variables[~model_variables['equation'].apply(lambda x: x.replace('.', '', 1).isdigit())]
            model_externals = model_externals.reset_index(drop=True)

            stocks = symbols(model_stocks.name.tolist())

            if len(model_parameters) > 0:
                parameter_symbols = symbols(model_parameters.name.tolist())
                args = np.array(model_parameters.equation, dtype=float)
            
            t = symbols('t')
            ode_text = ""
            
            for j in range(len(model_flows.FlowRate)):
                for i in range(len(model_externals)):
                    pattern = re.compile(re.escape(model_externals.name[i]))
                    model_flows.FlowRate[j] = pattern.sub(model_externals.equation[i], model_flows.FlowRate[j])

            stocks_dot = []
            for i in range(len(stocks)):
                d = 0
                for j in range(len(model_flows)):
                    if model_flows.target[j] == model_stocks.identity[i]:
                        d = d + sympify(model_flows.FlowRate[j])
                    elif model_flows.source[j] == model_stocks.identity[i]:
                        d = d - sympify(model_flows.FlowRate[j])
                stocks_dot.append(d)
            #for i in range(len(stocks)):
            #    print('d/dt(' f'{stocks[i]})' '=' f'{stocks_dot[i]}')
                ode_text += 'd/dt(' f'{stocks[i]})' '=' f'{stocks_dot[i]}\n'
            
            text_widget = tk.Text(self.master, wrap=tk.WORD, width=50, height=10)
            text_widget.insert(tk.END, ode_text)
            text_widget.pack(pady=20)
            

    def execute_model(self):
        if self.model_path:
            # Existing code for parsing and solving the Insightmaker model
            model_variables = parse_insightmaker_variables(self.model_path)
            model_stocks = parse_insightmaker_stocks(self.model_path)
            model_flows = parse_insightmaker_flows(self.model_path)
            simulation_settings = parse_insightmaker_settings(self.model_path)

            tspan = np.linspace(simulation_settings[0], simulation_settings[0] + simulation_settings[1],
                                1 + int(np.floor(simulation_settings[1] / simulation_settings[2])))
            X0 = model_stocks.initial_value

            model_parameters = model_variables[model_variables['equation'].apply(lambda x: x.replace('.', '', 1).isdigit())]
            model_parameters = model_parameters.reset_index(drop=True)

            model_externals = model_variables[~model_variables['equation'].apply(lambda x: x.replace('.', '', 1).isdigit())]
            model_externals = model_externals.reset_index(drop=True)

            stocks = symbols(model_stocks.name.tolist())

            if len(model_parameters) > 0:
                parameter_symbols = symbols(model_parameters.name.tolist())
                args = np.array(model_parameters.equation, dtype=float)
            
            t = symbols('t')
            for j in range(len(model_flows.FlowRate)):
                for i in range(len(model_externals)):
                    pattern = re.compile(re.escape(model_externals.name[i]))
                    model_flows.FlowRate[j] = pattern.sub(model_externals.equation[i], model_flows.FlowRate[j])

            stocks_dot = []
            for i in range(len(stocks)):
                d = 0
                for j in range(len(model_flows)):
                    if model_flows.target[j] == model_stocks.identity[i]:
                        d = d + sympify(model_flows.FlowRate[j])
                    elif model_flows.source[j] == model_stocks.identity[i]:
                        d = d - sympify(model_flows.FlowRate[j])
                stocks_dot.append(d)

            SolMethod = 'DOP853'

            if len(model_parameters) > 0:
                model_func = lambdify((t, stocks) + tuple(parameter_symbols), stocks_dot)
                solution = scipy.integrate.solve_ivp(model_func, (simulation_settings[0],simulation_settings[0] + simulation_settings[1]),X0, method=SolMethod, t_eval=tspan, args=args)
            else:
                model_func = lambdify((t, stocks), stocks_dot)
                solution = scipy.integrate.solve_ivp(model_func, (simulation_settings[0],simulation_settings[0] + simulation_settings[1]),X0, method=SolMethod, t_eval=tspan)

            # Plot the solutions
            self.plot_time_solutions(solution)
            
            if len(model_stocks) == 2:
                self.plot_phase_solutions(solution)

    def plot_time_solutions(self, solution):
        model_stocks = parse_insightmaker_stocks(self.model_path)
        fig, ax = plt.subplots()
        ax.plot(solution.t, solution.y.T)
        ax.set_xlabel('Time')
        ax.set_ylabel('Variables')
        ax.legend([model_stocks.name[i] for i in range(len(solution.y))])
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        
    def plot_phase_solutions(self, solution):
        model_stocks = parse_insightmaker_stocks(self.model_path)
        fig, ax = plt.subplots()
        ax.plot(solution.y[0], solution.y[1])
        ax.set_xlabel(model_stocks.name[0])
        ax.set_ylabel(model_stocks.name[1])
        #ax.legend([model_stocks.name[i] for i in range(len(solution.y))])
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelExecutorApp(root)
    root.mainloop()

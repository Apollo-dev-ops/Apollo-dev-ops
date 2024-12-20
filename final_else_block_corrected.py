import customtkinter as ctk
import numpy as np
import sympy as sp
from sympy.abc import x, y, z, t
from sympy import (
I, solve, Matrix, diff, integrate, limit, series, conjugate,
re, im, exp, sqrt, sin, cos, tan, asin, acos, atan, sinh,
cosh, tanh, pi, E, oo, Rational, Symbol, latex,
factorint, isprime, gcd, lcm, totient, mobius, divisors
)
from sympy.parsing.sympy_parser import transformations, parse_expr
import scipy.stats as stats
from scipy.optimize import fsolve, minimize
import scipy.special as special
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import partial
from typing import Any, Optional
import warnings
warnings.filterwarnings('ignore')
class AdvancedMathCalculator(ctk.CTk):
    def __init__(self):
        super().__init__()
        # Configure window
        self.title("Advanced Mathematical Calculator")
        self.attributes('-fullscreen', True)
        # Samsung A30 color scheme
        self.COLORS = {
        'background': "#121212",
        'display_bg': "#1E1E1E",
        'button_normal': "#2D2D2D",
        'button_special': "#313131",
        'button_operator': "#FF9500",
        'text_primary': "#FFFFFF",
        'text_secondary': "#B3B3B3",
        'accent': "#FF9500"
        }
        # Configure styling
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        # Initialize variables
        self.expression = ""
        self.history = []
        self.memory = 0
        self.last_answer = 0
        self.current_angle_mode = "RAD"  # RAD or DEG
        self.current_number_base = 10    # 2, 8, 10, 16
        # Create UI
        self.init_ui()
    def init_ui(self):
        # Main container
        self.main_frame =ctk.CTkFrame(self, fg_color=self.COLORS['background'])
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # Top bar with mode toggles
        self.create_top_bar()
        # History and display area
        self.create_display_area()
        # Tab view for different mathematical domains
        self.create_tab_view()
        # Create all specialized tabs
        self.create_basic_tab()
        self.create_advanced_tabs(
    def create_top_bar(self):
        top_bar = ctk.CTkFrame(self.main_frame, fg_color=self.COLORS['button_special'])
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        # Mode toggles
        modes = [
        ("RAD", "DEG"),
        ("F-E", ""),
        ("MC", "MR"),
        ]
        for col, (left_text, right_text) in enumerate(modes):
            frame = ctk.CTkFrame(top_bar, fg_color="transparent")
            frame.grid(row=0, column=col, padx=5, pady=2)
            # Create the left button if left_text is not empty
            if left_text:
                left_btn = ctk.CTkButton(
                frame,
                text=left_text,
                width=60,
                height=30,
                font=("Arial", 14),
                fg_color=self.COLORS['button_normal'],
                hover_color=self.COLORS['button_special'],
                command=partial(self.on_mode_toggle, left_text)
                )
                left_btn.grid(row=0, column=0, padx=2)
                # Create the right button if right_text is not empty
                if right_text:
                    right_btn = ctk.CTkButton(
                    frame,
                    text=right_text,
                    width=60,
                    height=30,
                    font=("Arial", 14),
                    fg_color=self.COLORS['button_normal'],
                    hover_color=self.COLORS['button_special'],
                    command=partial(self.on_mode_toggle, right_text)
                    )
                    right_btn.grid(row=0, column=1, padx=2)
    def create_display_area(self):
        display_frame = ctk.CTkFrame(self.main_frame, fg_color=self.COLORS['display_bg'])
        display_frame.grid(row=1, column=0, sticky="ew", pady=5)
        # History display
        self.history_display = ctk.CTkTextbox(
        display_frame,
        height=100,
        font=("Arial", 16),
        fg_color=self.COLORS['display_bg'],
        text_color=self.COLORS['text_secondary']
        )
        self.history_display.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        # Main display
        self.display = ctk.CTkEntry(
        display_frame,
        height=80,
        font=("Arial", 32),
        fg_color=self.COLORS['display_bg'],
        text_color=self.COLORS['text_primary'],
        justify="right"
        )
        self.display.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        # Result label
        self.result_label = ctk.CTkLabel(
        display_frame,
        text="",
        height=30,
        font=("Arial", 16),
        text_color=self.COLORS['text_secondary']
        )
        self.result_label.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
    def create_tab_view(self):
        self.tab_view = ctk.CTkTabview(self.main_frame, height=600)
        self.tab_view.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        # Create tabs
        self.tabs = {
        "Basic": self.tab_view.add("Basic"),
        "Functions": self.tab_view.add("Functions"),
        "Algebra": self.tab_view.add("Algebra"),
        "Calculus": self.tab_view.add("Calculus"),
        "Complex": self.tab_view.add("Complex"),
        "Matrix": self.tab_view.add("Matrix"),
        "Statistics": self.tab_view.add("Statistics"),
        "Number Theory": self.tab_view.add("Number Theory"),
        "Differential Eq": self.tab_view.add("Differential Eq"),
        "Graph": self.tab_view.add("Graph")
        }
    def create_basic_tab(self):
        basic_buttons = [
        ('(', ')', '%', '÷'),
        ('7', '8', '9', '×'),
        ('4', '5', '6', '−'),
        ('1', '2', '3', '+'),
        ('±', '0', '.', '=')
        ]
        special_buttons = [
        ('C', 'CE'),
        ('⌫', 'ANS'),
        ('x²', '√x'),
        ('1/x', '|x|'),
        ('EXIT', 'HELP')
        ]
        # Create main buttons
        for row, row_buttons in enumerate(basic_buttons):
            for col, text in enumerate(row_buttons):
                button = ctk.CTkButton(
                self.tabs["Basic"],
                text=text,
                width=80,
                height=60,
                font=("Arial", 24),
                fg_color=self.COLORS['button_operator'] if text in '÷×−+=' else self.COLORS['button_normal'],
                hover_color=self.COLORS['button_special'],
                command=lambda t=text: self.on_button_click(t)
                )
                button.grid(row=row, column=col+1, padx=2, pady=2)
                # Create special buttons column
                for row, (top_text, bottom_text) in enumerate(special_buttons):
                    for col, text in enumerate([top_text, bottom_text]):
                        button = ctk.CTkButton(
                        self.tabs["Basic"],
                        text=text,
                        width=80,
                        height=60,
                        font=("Arial", 20),
                        fg_color=self.COLORS['button_special'],
                        hover_color=self.COLORS['button_normal'],
                        command=lambda t=text: self.on_button_click(t)
                        )
                        button.grid(row=row, column=col, padx=2, pady=2)
    def create_advanced_tabs(self):
        # Functions tab
        function_buttons = [
        ('sin', 'cos', 'tan', 'ln', 'log'),
        ('asin', 'acos', 'atan', 'e^x', '10^x'),
        ('sinh', 'cosh', 'tanh', 'π', 'e'),
        ('deg', 'rad', 'grad', '∞', 'rand')
        ]
        self.create_function_buttons(self.tabs["Functions"], function_buttons)
        # Algebra tab
        algebra_buttons = [
        ('Solve', 'Factor', 'Expand', 'Simplify', 'GCD'),
        ('LCM', 'Mod', 'Floor', 'Ceil', 'Round'),
        ('Polynomial', 'Rational', 'Linear Eq', 'Quadratic', 'Cubic'),
        ('Matrix', 'Vector', 'Group', 'Ring', 'Field')
        ]
        self.create_function_buttons(self.tabs["Algebra"], algebra_buttons)
        # Calculus tab
        calculus_buttons = [
        ('∫ Indefinite', '∫ Definite', 'd/dx', 'd²/dx²', '∂/∂x'),
        ('Limit', 'Series', 'Taylor', 'Maclaurin', 'Power'),
        ('Line Int', 'Surface Int', 'Volume Int', 'Curl', 'Div'),
        ('Gradient', 'Laplacian', 'Jacobian', 'Hessian', 'Extrema')
        ]
        self.create_function_buttons(self.tabs["Calculus"], calculus_buttons)
        # Complex tab
        complex_buttons = [
        ('i', 'Conjugate', 'Abs', 'Arg', 'Phase'),
        ('Re', 'Im', 'Polar', 'Rect', 'Euler'),
        ('exp(iθ)', 'ln(z)', 'Power', 'Root', 'Unit'),
        ('Complex Int', 'Residue', 'Laurent', 'Conformal', 'Möbius')
        ]
        self.create_function_buttons(self.tabs["Complex"], complex_buttons)
        self.create_matrix_tab()
        self.create_statistics_tab()
        self.create_number_theory_tab()
        self.create_differential_equations_tab()
        self.create_graph_tab()
    def create_function_buttons(self, tab, buttons):
        for row, row_buttons in enumerate(buttons):
            for col, text in enumerate(row_buttons):
                button = ctk.CTkButton(
                tab,
                text=text,
                width=100,
                height=60,
                font=("Arial", 16),
                fg_color=self.COLORS['button_normal'],
                hover_color=self.COLORS['button_special'],
                command=lambda t=text: self.on_function_click(t)
                )
                button.grid(row=row, column=col, padx=2, pady=2)
    def on_button_click(self, button_text: str) -> None:
        if button_text == "EXIT":
            self.quit()
        elif button_text == "=":
                self.evaluate_expression()
        elif button_text == "C":
                    self.clear_all()
        elif button_text == "CE":
                        self.clear_entry()
        elif button_text == "⌫":
                            self.backspace()
        elif button_text == "ANS":
                                self.use_last_answer()
                else:
                    self.append_to_expression(button_text)
    def evaluate_expression(self) -> None:
        try:
            # Replace visual operators with computational ones
            expr = self.expression.replace('×', '*').replace('÷', '/').replace('−', '-')
            # Handle special cases (percentages, etc.)
            if '%' in expr:
                expr = self.handle_percentage(expr)
                # Convert to correct angle mode if needed
                if any(trig_func in expr for trig_func in ['sin', 'cos', 'tan']):
                    expr = self.convert_angle_mode(expr)
                    # Evaluate using sympy
                    result = sp.sympify(expr).evalf()
                    # Format result
                    if result is not None and isinstance(result, (sp.Integer, sp.Float)):
                        formatted_result = f"{float(result):g}"
                else:
                    formatted_result = str(result)
                            # Update displays
                            self.last_answer = result
                            self.history.append(f"{self.expression} = {formatted_result}")
                            self.update_history()
                            self.result_label.configure(text=f"= {formatted_result}")
                            self.display.delete(0, 'end')
                            self.display.insert(0, formatted_result)
                            self.expression = formatted_result
                            except ValueError as e:
                                self.result_label.configure(text=f"Error: {str(e)}")
    def handle_percentage(self, expr: str) -> str:
        # Implement percentage calculations
        parts = expr.split('%')
        if len(parts) != 2:
            raise ValueError("Invalid percentage expression")
            base = float(parts[0])
            return str(base / 100)
    def convert_angle_mode(self, expr: str) -> str:
        if self.current_angle_mode == "DEG":
            # Convert degrees to radians for calculation
            expr = expr.replace('sin(', 'sin(pi/180*')
            expr = expr.replace('cos(', 'cos(pi/180*')
            expr = expr.replace('tan(', 'tan(pi/180*')
            return expr
    def create_matrix_tab(self):
        matrix_frame = ctk.CTkFrame(self.tabs["Matrix"])
        matrix_frame.grid(sticky="nsew", padx=5, pady=5)
        # Matrix size controls
        size_frame = ctk.CTkFrame(matrix_frame)
        size_frame.grid(row=0, column=0, columnspan=2, pady=5)
        ctk.CTkLabel(size_frame, text="Matrix Size:").grid(row=0, column=0)
        self.rows_var = ctk.StringVar(value="2")
        self.cols_var = ctk.StringVar(value="2")
        ctk.CTkEntry(size_frame, textvariable=self.rows_var, width=50).grid(
        row=0, column=1
        )
        ctk.CTkLabel(size_frame, text="×").grid(row=0, column=2)
        ctk.CTkEntry(size_frame, textvariable=self.cols_var, width=50).grid(
        row=0, column=3
        )
        # Matrix input area
        self.matrix_entries = []
        self.matrix_frame = ctk.CTkFrame(matrix_frame)
        self.matrix_frame.grid(row=1, column=0, pady=5)
        self.update_matrix_size()
        # Matrix operations
        operations = [
        ("Determinant", self.calculate_determinant),
        ("Inverse", self.calculate_inverse),
        ("Transpose", self.calculate_transpose),
        ("Eigenvalues", self.calculate_eigenvalues),
        ("⌫", self.backspace),
        ]
        ops_frame = ctk.CTkFrame(matrix_frame)
        ops_frame.grid(row=1, column=1, padx=5)
        for i, (text, command) in enumerate(operations):
            ctk.CTkButton(
            ops_frame, text=text, command=command, width=100, height=40
            ).grid(row=i, column=0, pady=2)
            # Create plotting area
            self.figure_frame = ctk.CTkFrame(matrix_frame)
            self.figure_frame.grid(row=3, column=0, columnspan=2, pady=5)
            self.fig, self.ax = plt.subplots(figsize=(6, 4))  # Create Figure and Axes
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=0, column=0)
    def update_matrix_size(self):
        # Clear existing entries
        for row in self.matrix_entries:
            for entry in row:
                entry.destroy()
                self.matrix_entries.clear()
                # Create new entries
                rows = int(self.rows_var.get())
                cols = int(self.cols_var.get())
                for i in range(rows):
                    row_entries = []
                    for j in range(cols):
                        entry = ctk.CTkEntry(self.matrix_frame, width=50)
                        entry.grid(row=i, column=j, padx=2, pady=2)
                        row_entries.append(entry)
                        self.matrix_entries.append(row_entries)
    def get_matrix(self):
        if not self.matrix_entries or not self.matrix_entries[0]:
            self.result_label.configure(text="Error: Invalid matrix dimensions")
            raise ValueError('Invalid matrix dimensions or entries')
            rows = len(self.matrix_entries)
            cols = len(self.matrix_entries[0])
            matrix = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    try:
                        value = float(self.matrix_entries[i][j].get())
                        row.append(value)
                        except ValueError:
                            self.result_label.configure(text="Error: Invalid matrix entry")
                            raise ValueError('Invalid matrix dimensions or entries')
                            matrix.append(row)
                            return Matrix(matrix)
    def calculate_determinant(self):
        matrix = self.get_matrix()
        if matrix is not None:
            try:
                det = matrix.det()
                self.result_label.configure(text=f"Determinant = {det}")
                except ValueError:
                    self.result_label.configure(text="Error: Cannot calculate determinant")
    def calculate_inverse(self):
        matrix = self.get_matrix()
        if matrix is not None:
            try:
                inv = matrix.inv()
                self.display_matrix_result(inv)
                except ValueError:
                    self.result_label.configure(text="Error: Matrix is not invertible")
    def calculate_transpose(self):
        matrix = self.get_matrix()
        if matrix is not None:
            try:
                trans = matrix.transpose()
                self.display_matrix_result(trans)
                except ValueError:
                    self.result_label.configure(text="Error: Cannot calculate transpose")
    def calculate_eigenvalues(self):
        matrix = self.get_matrix()
        if matrix is not None:
            try:
                eigvals = matrix.eigenvals()
                self.result_label.configure(text=f"Eigenvalues = {eigvals}")
                except ValueError:
                    self.result_label.configure(text="Error: Cannot calculate eigenvalues")
    def display_matrix_result(self, matrix):
        result_window = ctk.CTkToplevel(self)
        result_window.title("Matrix Result")
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                ctk.CTkLabel(result_window, text=f"{matrix[i, j]:.4f}").grid(
                row=i, column=j, padx=5, pady=5
                )
    def create_statistics_tab(self):
        stats_frame = ctk.CTkFrame(self.tabs["Statistics"])
        stats_frame.grid(sticky="nsew", padx=5, pady=5)
        # Data input
        input_frame = ctk.CTkFrame(stats_frame)
        input_frame.grid(row=0, column=0, columnspan=2, pady=5)
        ctk.CTkLabel(input_frame, text="Enter data (comma-separated):").grid(
        row=0, column=0
        )
        self.data_entry = ctk.CTkEntry(input_frame, width=300)
        self.data_entry.grid(row=0, column=1, padx=5)
        # Statistical operations
        operations = [
        ("Mean", self.calculate_mean),
        ("Median", self.calculate_median),
        ("Mode", self.calculate_mode),
        ("Std Dev", self.calculate_std_dev),
        ("Variance", self.calculate_variance),
        ("Range", self.calculate_range),
        ("Quartiles", self.calculate_quartiles),
        ("Skewness", self.calculate_skewness),
        ("Kurtosis", self.calculate_kurtosis),
        ("⌫", self.backspace),
        ]
        for i, (text, command) in enumerate(operations):
            ctk.CTkButton(
            stats_frame, text=text, command=command, width=100, height=40
            ).grid(row=(i // 2) + 1, column=i % 2, padx=5, pady=2)
    def get_data(self):
        try:
            data_str = self.data_entry.get()
            return [float(x.strip()) for x in data_str.split(",")]
            except ValueError:
                self.result_label.configure(text="Error: Invalid data format")
                raise ValueError('Invalid matrix dimensions or entries')
    def calculate_mean(self):
        data = self.get_data()
        if data:
            mean = np.mean(data)
            self.result_label.configure(text=f"Mean = {mean:.4f}")
    def calculate_median(self):
        data = self.get_data()
        if data:
            median = np.median(data)
            self.result_label.configure(text=f"Median = {median:.4f}")
    def calculate_mode(self):
        data = self.get_data()
        if data:
            mode = stats.mode(data)
            self.result_label.configure(text=f"Mode = {mode.mode[0]:.4f}")
    def calculate_std_dev(self):
        data = self.get_data()
        if data:
            std = np.std(data)
            self.result_label.configure(text=f"Standard Deviation = {std:.4f}")
    def calculate_variance(self):
        data = self.get_data()
        if data:
            var = np.var(data)
            self.result_label.configure(text=f"Variance = {var:.4f}")
    def calculate_range(self):
        data = self.get_data()
        if data:
            range_val = max(data) - min(data)
            self.result_label.configure(text=f"Range = {range_val:.4f}")
    def calculate_quartiles(self):
        data = self.get_data()
        if data:
            q1, q2, q3 = np.percentile(data, [25, 50, 75])
            self.result_label.configure(text=f"Q1={q1:.4f}, Q2={q2:.4f}, Q3=             {q3:.4f}")
    def calculate_skewness(self):
        data = self.get_data()
        if data:
            skew = stats.skew(data)
            self.result_label.configure(text=f"Skewness = {skew:.4f}")
    def calculate_kurtosis(self):
        data = self.get_data()
        if data:
            kurt = stats.kurtosis(data)
            self.result_label.configure(text=f"Kurtosis = {kurt:.4f}")
    def create_number_theory_tab(self):
        nt_frame = ctk.CTkFrame(self.tabs["Number Theory"])
        nt_frame.grid(sticky="nsew", padx=5, pady=5)
        # Number input
        input_frame = ctk.CTkFrame(nt_frame)
        input_frame.grid(row=0, column=0, columnspan=2, pady=5)
        ctk.CTkLabel(input_frame, text="Enter number:").grid(row=0, column=0)
        self.number_entry = ctk.CTkEntry(input_frame, width=200)
        self.number_entry.grid(row=0, column=1, padx=5)
        # Second number input for operations requiring two numbers
        ctk.CTkLabel(input_frame, text="Second number (if needed):").grid(
        row=1, column=0
        )
        self.number_entry2 = ctk.CTkEntry(input_frame, width=200)
        self.number_entry2.grid(row=1, column=1, padx=5)
        # Number theory operations
        operations = [
        ("Prime Factors", self.prime_factorization),
        ("Is Prime?", self.is_prime),
        ("GCD", self.calculate_gcd),
        ("LCM", self.calculate_lcm),
        ("Euler's Totient", self.euler_totient),
        ("Möbius", self.mobius),
        ("Divisors", self.find_divisors),
        ("Primitive Root", self.primitive_root),
        ("Legendre Symbol", self.legendre_symbol),
        ("⌫", self.backspace),
        ]
        for i, (text, command) in enumerate(operations):
            ctk.CTkButton(
            nt_frame, text=text, command=command, width=100, height=40
            ).grid(row=(i // 2) + 2, column=i % 2, padx=5, pady=2)
    def get_number(self):
        try:
            return int(self.number_entry.get())
            except ValueError:
                self.result_label.configure(text="Error: Invalid number format")
                raise ValueError('Invalid matrix dimensions or entries')
    def get_second_number(self):
        try:
            return int(self.number_entry2.get())
            except ValueError:
                self.result_label.configure(text="Error: Invalid second number format")
                raise ValueError('Invalid matrix dimensions or entries')
    def prime_factorization(self):
        n = self.get_number()
        if n:
            factors = factorint(n)
            result = " × ".join(
            [f"{p}^{e}" if e > 1 else str(p) for p, e in factors.items()]
            )
            self.result_label.configure(text=f"Prime factorization: {result}")
    def is_prime(self):
        n = self.get_number()
        if n is not None:
            result = isprime(n)
            self.result_label.configure(
            text=f"{n} is {'prime' if result else 'not prime'}"
            )
    def calculate_gcd(self):
        a = self.get_number()
        b = self.get_second_number()
        if a and b:
            result = gcd(a, b)
            self.result_label.configure(text=f"GCD({a}, {b}) = {result}")
    def calculate_lcm(self):
        a = self.get_number()
        b = self.get_second_number()
        if a and b:
            result = lcm(a, b)
            self.result_label.configure(text=f"LCM({a}, {b}) = {result}")
    def euler_totient(self):
        n = self.get_number()
        if n:
            result = totient(n)
            self.result_label.configure(text=f"φ({n}) = {result}")
    def mobius(self):
        n = self.get_number()
        if n:
            result = mobius(n)
            self.result_label.configure(text=f"μ({n}) = {result}")
    def find_divisors(self):
        n = self.get_number()
        if n is not None:
            divs = divisors(n)
            self.result_label.configure(text=f"Divisors of {n}: {sorted(divs)}")
    def primitive_root(self):
        n = self.get_number()
        if n:
            try:
                root = primitive_root(n)
                self.result_label.configure(
                text=f"Smallest primitive root of {n}: {root}"
                )
                except ValueError:
                    self.result_label.configure(text=f"No primitive root exists for {n}")
    def legendre_symbol(self):
        a = self.get_number()
        p = self.get_second_number()
        if a and p:
            result = legendre_symbol(a, p)
            self.result_label.configure(text=f"({a}/{p}) = {result}")
    def create_differential_equations_tab(self):
        de_frame = ctk.CTkFrame(self.tabs["Differential Eq"])
        de_frame.grid(sticky="nsew", padx=5, pady=5)
        # Equation input
        input_frame = ctk.CTkFrame(de_frame)
        input_frame.grid(row=0, column=0, columnspan=2, pady=5)
        ctk.CTkLabel(input_frame, text="Enter differential equation:").grid(
        row=0, column=0
        )
        self.de_entry = ctk.CTkEntry(input_frame, width=300)
        self.de_entry.grid(row=0, column=1, padx=5)
        # Initial conditions
        ctk.CTkLabel(input_frame, text="Initial conditions (y(0)=...):").grid(
        row=1, column=0
        )
        self.ic_entry = ctk.CTkEntry(input_frame, width=300)
        self.ic_entry.grid(row=1, column=1, padx=5)
        # DE operations
        operations = [
        ("Solve ODE", self.solve_ode),
        ("Phase Plot", self.phase_plot),
        ("Numerical Solution", self.numerical_solution),
        ("Stability Analysis", self.stability_analysis),
        ("Equilibrium Points", self.equilibrium_points),
        ("Bifurcation Analysis", self.bifurcation_analysis),
        ("System of DEs", self.system_of_des),
        ("Boundary Value", self.boundary_value),
        ("Parameter Study", self.parameter_study),
        ("⌫", self.backspace),
        ]
        for i, (text, command) in enumerate(operations):
            ctk.CTkButton(
            de_frame, text=text, command=command, width=120, height=40
            ).grid(row=(i // 2) + 2, column=i % 2, padx=5, pady=2)
    def solve_ode(self):
        try:
            eq = self.de_entry.get()
            ic = self.ic_entry.get()
            # Use sympy to solve the ODE symbolically
            t = Symbol("t")
            y = Function("y")
            eq_parsed = parse_expr(eq)
            sol = dsolve(eq_parsed, y(t))
            self.result_label.configure(text=f"Solution: {sol}")
            except ValueError as e:
                self.result_label.configure(text=f"Error: {str(e)}")
    def create_graph_tab(self):
        graph_frame = ctk.CTkFrame(self.tabs["Graph"])
        graph_frame.grid(sticky="nsew", padx=5, pady=5)
        # Input controls
        controls_frame = ctk.CTkFrame(graph_frame)
        controls_frame.grid(row=0, column=0, columnspan=2, pady=5)
        # Function input
        ctk.CTkLabel(controls_frame, text="f(x) = ").grid(row=0, column=0)
        self.function_entry = ctk.CTkEntry(controls_frame, width=300)
        self.function_entry.grid(row=0, column=1, padx=5)
        self.function_entry.insert(0, "x**2")  # Default function
        # Range inputs
        range_frame = ctk.CTkFrame(controls_frame)
        range_frame.grid(row=1, column=0, columnspan=2, pady=5)
        ctk.CTkLabel(range_frame, text="x range:").grid(row=0, column=0)
        self.x_min = ctk.CTkEntry(range_frame, width=50)
        self.x_min.insert(0, "-10")
        self.x_min.grid(row=0, column=1)
        ctk.CTkLabel(range_frame, text="to").grid(row=0, column=2)
        self.x_max = ctk.CTkEntry(range_frame, width=50)
        self.x_max.insert(0, "10")
        self.x_max.grid(row=0, column=3)
        # Plot controls
        plot_controls = [
        ("Plot", self.plot_function),
        ("Clear", self.clear_plot),
        ("Add Plot", self.add_plot),
        ("Grid", self.toggle_grid),
        ("Save", self.save_plot),
        ("Zoom In", self.zoom_in),
        ("Zoom Out", self.zoom_out),
        ("Reset View", self.reset_view),
        ("⌫", self.backspace),
        ]
        controls_frame = ctk.CTkFrame(graph_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=5)
        for i, (text, command) in enumerate(plot_controls):
            ctk.CTkButton(
            controls_frame, text=text, command=command, width=80, height=30
            ).grid(row=i // 3, column=i % 3, padx=2, pady=2)
            # Create plotting area
            self.figure_frame = ctk.CTkFrame(graph_frame)
            self.figure_frame.grid(row=3, column=0, columnspan=2, pady=5)
            if not hasattr(self, 'canvas'):
                self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().grid(row=0, column=0)
    def plot_function(self):
        try:
            # Clear previous plot
            self.ax.clear()
            # Get function and range
            func_str = self.function_entry.get()
            x_min = float(self.x_min.get())
            x_max = float(self.x_max.get())
            # Create x values
            x_vals = np.linspace(x_min, x_max, 1000)
            # Parse and evaluate function
            x = Symbol("x")
            func = parse_expr(func_str)
            y_vals = [float(func.subs(x, val)) for val in x_vals]
            # Plot
            self.ax.plot(x_vals, y_vals)
            self.ax.grid(True)
            self.ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            self.ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)
            # Update canvas
            self.canvas.draw()
            except ValueError as e:
                self.result_label.configure(text=f"Error: {str(e)}")
    def clear_plot(self):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        self.ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)
        self.canvas.draw()
    def add_plot(self):
        try:
            # Get function and range
            func_str = self.function_entry.get()
            x_min = float(self.x_min.get())
            x_max = float(self.x_max.get())
            # Create x values
            x_vals = np.linspace(x_min, x_max, 1000)
            # Parse and evaluate function
            x = Symbol("x")
            func = parse_expr(func_str)
            y_vals = [float(func.subs(x, val)) for val in x_vals]
            # Add to existing plot
            self.ax.plot(x_vals, y_vals)
            self.canvas.draw()
            except ValueError as e:
                self.result_label.configure(text=f"Error: {str(e)}")
    def toggle_grid(self):
        self.ax.grid(not self.ax._gridOnMajor)
        self.canvas.draw()
    def save_plot(self):
        try:
            self.fig.savefig("plot.png")
            self.result_label.configure(text="Plot saved as 'plot.png'")
            except ValueError as e:
                self.result_label.configure(text=f"Error saving plot: {str(e)}")
    def zoom_in(self):
        self.ax.set_xlim(self.ax.get_xlim()[0] * 0.8, self.ax.get_xlim()[1] * 0.8)
        self.ax.set_ylim(self.ax.get_ylim()[0] * 0.8, self.ax.get_ylim()[1] * 0.8)
        self.canvas.draw()
    def zoom_out(self):
        self.ax.set_xlim(self.ax.get_xlim()[0] * 1.2, self.ax.get_xlim()[1] * 1.2)
        self.ax.set_ylim(self.ax.get_ylim()[0] * 1.2, self.ax.get_ylim()[1] * 1.2)
        self.canvas.draw()
    def reset_view(self):
        x_min = float(self.x_min.get())
        x_max = float(self.x_max.get())
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(-10, 10)  # Default y range
        self.canvas.draw()
    def append_to_expression(self, text: str) -> None:
        """Append text to the current expression and update display."""
        self.expression += text
        self.display.delete(0, 'end')
        self.display.insert(0, self.expression)
    def clear_all(self) -> None:
        """Clear all expressions and results."""
        self.expression = ""
        self.display.delete(0, 'end')
        self.result_label.configure(text="")
    def clear_entry(self) -> None:
        """Clear only the current entry."""
        self.expression = ""
        self.display.delete(0, 'end')
    def backspace(self) -> None:
        """Remove the last character from the expression."""
        self.expression = self.expression[:-1]
        self.display.delete(0, 'end')
        self.display.insert(0, self.expression)
    def use_last_answer(self) -> None:
        """Insert the last calculated answer into the expression."""
        self.expression += str(self.last_answer)
        self.display.delete(0, 'end')
        self.display.insert(0, self.expression)
    def update_history(self) -> None:
        """Update the history display with the latest calculations."""
        self.history_display.delete('1.0', 'end')
        for item in self.history[-5:]:  # Show last 5 calculations
        self.history_display.insert('end', f"{item}\n")
    def on_mode_toggle(self, mode: str) -> None:
        """Handle mode toggle button clicks."""
        if mode in ["RAD", "DEG"]:
            self.current_angle_mode = mode
        elif mode == "F-E":
                # Toggle between fixed and scientific notation
                pass
        elif mode in ["MC", "MR"]:
                    self.handle_memory(mode)
    def on_function_click(self, function: str) -> None:
        """Handle clicks on advanced function buttons."""
        try:
            if not self.expression:
                return
                expr = parse_expr(self.display.get())
                result = None
                # Basic Functions
                if function in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']:
                    angle = expr if self.current_angle_mode == "RAD" else expr * pi / 180
                    result = getattr(sp, function)(angle)
                    # Hyperbolic Functions
        elif function in ['sinh', 'cosh', 'tanh']:
                        result = getattr(sp, function)(expr)
                        # Logarithmic Functions
        elif function == 'ln':
                            result = sp.log(expr)
        elif function == 'log':
                                result = sp.log(expr, 10)
                                # Exponential Functions
        elif function == 'e^x':
                                    result = sp.exp(expr)
        elif function == '10^x':
                                        result = 10**expr
                                        # Constants
        elif function == 'π':
                                            result = pi
        elif function == 'e':
                                                result = E
        elif function == '∞':
                                                    result = oo
                                                    # Format and display result
                                                    if result is not None:
                                                        formatted_result = self.format_result(result)
                                                        self.history.append(f"{function}({expr}) = {formatted_result}")
                                                        self.update_history()
                                                        self.display.delete(0, 'end')
                                                        self.display.insert(0, str(formatted_result))
                                                        self.expression = str(formatted_result)
                                                        except ValueError as e:
                                                            self.result_label.configure(text=f"Error: {str(e)}")
    def format_result(self, result: Any) -> str:
        """Format the calculation result for display."""
        if isinstance(result, (sp.Integer, sp.Float)):
            if abs(result) > 1e12 or (abs(result) < 1e-12 and result != 0):
                return f"{float(result):e}"
                return f"{float(result):g}"
                return str(result)
    def handle_memory(self, operation: str) -> None:
        """Handle memory operations."""
        try:
            if operation == "MC":
                self.memory = 0
        elif operation == "MR":
                    self.expression = str(self.memory)
                    self.display.delete(0, 'end')
                    self.display.insert(0, self.expression)
        elif operation == "M+":
                        self.memory += float(self.expression)
        elif operation == "M-":
                            self.memory -= float(self.expression)
        elif operation == "MS":
                                self.memory = float(self.expression)
                                except ValueError:
                                    self.result_label.configure(text="Error: Invalid expression for memory operation")
                                    if __name__ == "__main__":
                                        app = AdvancedMathCalculator()
                                        app.mainloop()

from environments.env import Environment


class ABCEnv(Environment):
    def __init__(self, equation: str):
        """
        Initializes the environment with 3 variables
        """
        # Initialize parent class
        super().__init__()

        # Initialize variables using a dictionary
        self.variables = {
            "A": 0,
            "B": 0,
            "C": 0,
            "D": "-",
        }

        self.input_variables = ["A", "B", "C"]
        self.output_variable = "D"

        self.equation = self.process_equation(equation)

    def process_equation(self, equation: str):
        """
        Processes the equation provided by the user and replaces the variables with their corresponding values.

        Parameters:
        equation (str): The equation provided by the user.

        Returns:
        str: The processed equation with variables replaced by their values.
        """
        for var in self.input_variables:
            equation = equation.replace(f"{var}^2", (f"{var} * {var}"))
        return equation

    def prepare_eval_equation(self, equation: str):
        """
        Prepares the equation for evaluation by replacing the squared variables.

        Parameters:
        equation (str): The equation provided by the user.

        Returns:
        str: The equation with variables replaced by their keys.
        """
        for var in self.input_variables:
            equation = equation.replace(var, str(self.variables[var]))
        return equation

    def calculate(self):
        """
        Calculates the output variable based on the equation provided.

        Returns:
        dict: The updated variables.
        """
        A = self.variables["A"]
        B = self.variables["B"]
        C = self.variables["C"]

        try:
            eq = self.prepare_eval_equation(self.equation)
            D = eval(eq)
        except ZeroDivisionError:
            D = "None"
        except Exception as e:
            D = "Error"

        # Update the output variable
        if not isinstance(D, str):
            self.variables[self.output_variable] = round(D, 3)
        else:
            self.variables[self.output_variable] = D

        return self.variables

    def set(self, var_name: str, value: float):
        """
        Sets the value of a variable and recalculates the output variable.

        Parameters:
        var_name (str): The name of the variable to set.
        value (float): The value to set the variable to.

        Returns:
        dict: The updated variables after recalculation.
        """
        if var_name in self.input_variables:
            self.variables[var_name] = value
            return self.calculate()
        else:
            raise ValueError(f"Variable {var_name} does not exist.")

    def getProcessEquation(self):
        """
        Returns the process equation.

        Returns:
        str: The equation as a formatted string.
        """
        return f"{self.output_variable} = {self.equation}"

    def relation(self, init_var, res_var: str):
        raise NotImplementedError(
            "Relation method is not implemented for this environment."
        )

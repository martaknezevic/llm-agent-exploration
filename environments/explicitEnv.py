from environments.env import Environment
import re

class ExplicitEnv(Environment):
    def __init__(self, equation: str):
        """
        Initializes the environment with 3 variables
        """
        # Initialize parent class
        super().__init__()

        # Initialize variables using a dictionary
        

        self.equation = equation
        self.output_variable, self.input_variables = self.process_equation(equation)
        
        self.variables = {var: 0 for var in self.input_variables} | {self.output_variable: '-'}

    def process_equation(self, equation: str):
        """
        Extracts the dependent and independent variables from a given equation.
        :param equation: A string representing an equation (e.g., 'foo = bar + lan / har')
        :return: A tuple (dependent_variable, independent_variables)
        """
        # Split equation at '='
        parts = equation.split('=')
        if len(parts) != 2:
            raise ValueError("Invalid equation format")
        
        # Extract dependent variable (left-hand side of '=')
        dependent_variable = parts[0].strip()
        
        # Extract independent variables from the right-hand side
        rhs = parts[1]
        independent_variables = set(re.findall(r'\b[a-zA-Z_]\w*\b', rhs))
        
        # Remove function names and operators if needed (assuming no function calls in input)
        
        return dependent_variable, sorted(independent_variables)

    def prepare_eval_equation(self):
        """
        Prepares the equation for evaluation by replacing the squared variables.

        Parameters:
        equation (str): The equation provided by the user.

        Returns:
        str: The equation with variables replaced by their keys.
        """
        
        calc_equation = self.equation.split('=')[1].strip()
        for var in self.input_variables:
            calc_equation = calc_equation.replace(var, str(self.variables[var]),1)
        return calc_equation

    def calculate(self):
        """
        Calculates the output variable based on the equation provided.

        Returns:
        dict: The updated variables.
        """

        try:
            eq = self.prepare_eval_equation()
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
        return f"{self.equation}"

    def relation(self, init_var, res_var: str):
        raise NotImplementedError(
            "Relation method is not implemented for this environment."
        )

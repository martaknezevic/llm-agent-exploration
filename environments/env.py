import re
from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self) -> None:
        self.variables: dict[str, int] = {}
        self.mapping: dict[str, str] = {}

        self.done = False

    def step(self, action: str):
        if action.startswith("set"):
            vars = re.search(r"\((.*)\)", action).group(1).split(",")
            
            res = None
            for var in vars:
                # using re extract variable name
                match = re.search(r"(\w+) *= *(-?\d+(\.\d+)?)", var)

                if match:
                    variable = match.group(1)#.upper()
                    value = float(match.group(2))
                    value = int(value) if value.is_integer() else value

                    # TODO : clean this up
                    if self.__class__.__name__ == "ABCEnv" or self.__class__.__name__ == "ExplicitEnv":
                        function_name = f"set"
                        if hasattr(self, function_name):
                            try: 
                                res = getattr(self, function_name)(variable, value)
                            except Exception as e:
                                raise e
                    else:
                        function_name = f"set{variable}"
                        if hasattr(self, function_name):
                            try: 
                                res = getattr(self, function_name)(value)
                            except Exception as e:
                                raise e
                            
            return res, self.done, None, vars
                
        elif action.startswith("finish"):
            self.done = True
            action = action.split('#')[0].strip()
            match = re.search(r"finish\((.*?)\)[^\)]*$", action)
            if match:
                model = match.group(1)
            return None, self.done, model, None

    def evaluate(self, calc_var, variables: str):
        vars = variables.split(";")

        for var in vars:
            var_name, var_value = var.split("=")

            function_name = f"set{var_name}"
            if hasattr(self, function_name):
                getattr(self, function_name)(int(var_value))

        return self.variables[calc_var]

    def get_variable_mapping(self):
        return self.mapping

    @abstractmethod
    def relation(self, init_var, res_var: str):
        pass

    @abstractmethod
    def getProcessEquation(self):
        """
        Returns the linear equation based on current constants and variables.

        Returns:
        str: The equation as a formatted string representing y = a * A + b.
        """
        pass

import os


class Prompt:
    def __init__(self, config):
        # Task parameters
        self.inquiry = config.task.inquiry
        self.memory = config.task.memory
        self.think = config.task.think
        self.explicit_modeling = config.task.explicit_model
        self.variable_ranges = config.env.variable_ranges
        self.zero_shot = config.task.zero_shot
        self.hard_aot = config.task.hard_aot

        # Files
        self.example_file = config.task.example_file
        self.env_file = config.env.env_file
        self.instruction_file = config.task.instruction_file

        self.initial_prompt = ""
        self.prompt = ""
        self.dir = config.task.prompt_dir

    def load_template(self, filename):
        file_path = os.path.join(self.dir, filename)

        if os.path.exists(file_path):
            with open(file_path) as f:
                return f.read()
        else:
            raise ValueError(f"File {file_path} does not exist.")

    def filter_lines(self, input_string, filter_string):
        """
        Splits the input string by newlines, filters out lines that start with filter_string,
        and returns the remaining lines as a new string.

        Parameters:
        input_string (str): The string to process.
        filter_string (str): String used for filtering

        Returns:
        str: The filtered string with lines not starting with filter_string.
        """
        lines = input_string.split("\n")

        filtered_lines = [line for line in lines if not line.startswith(filter_string)]

        return "\n".join(filtered_lines)

    def create_range_template(self):
        ranges = ""
        if self.variable_ranges:
            ranges = ""
            for key, value in self.variable_ranges.items():
                min_val = value[0]
                max_val = value[1]

                if min_val != None and max_val != None:
                    ranges += f"Variable {key} can take values between {value[0]} and {value[1]}\n"
                elif min_val != None:
                    ranges += (
                        f"Variable {key} can take values greater than {value[0]}\n"
                    )
                elif max_val != None:
                    ranges += f"Variable {key} can take values less than {value[1]}\n"
                else:
                    ranges += ""
        return ranges

    def build_initial_prompt(self):
        self.initial_prompt = self.load_template(self.instruction_file)
        env_description = self.load_template(self.env_file)
        variable_ranges = self.create_range_template()

        if self.zero_shot:
            examples = ""
        else:
            examples = self.load_template(self.example_file)

        think = ""
        memory = ""
        steps = ""
        think_instr2 = ""

        if self.explicit_modeling:
            model_task = " and model"
            model_return_arg = "<model>"
            model_return = " and returns modeled equation"
        else:
            model_task = ""
            model_return_arg = ""
            model_return = ""

        if self.think:
            think = self.load_template("instr_thinking.txt")
            steps += ", Thought"
            think_instr2 = " After every new observation, use the Thought step to analyse it and reason about the observed change. Do not use multiple Thought steps in a sequence."
        else:
            examples = self.filter_lines(examples, "Thought:")

        if self.memory:
            memory = self.load_template("instr_memory.txt")
            steps += ", Memory"
        else:
            examples = self.filter_lines(examples, "Memory:")

        steps += " and Observation"

        if self.inquiry:
            self.initial_prompt = self.initial_prompt.format(
                model_task,
                model_return_arg,
                model_return,
                examples,
                env_description,
                variable_ranges,
            )
        else:
            self.initial_prompt = self.initial_prompt.format(
                model_task,
                steps,
                model_return_arg,
                model_return,
                think,
                memory,
                examples,
                env_description,
                variable_ranges,
                think_instr2,
            )
        self.prompt = self.initial_prompt
        return self.initial_prompt

    def add_to_prompt(self, update):
        self.prompt += update

    def build_prompt(self, agent_history):
        self.prompt = self.initial_prompt + agent_history
        return self.prompt

    def get_prompt(self):
        return self.prompt

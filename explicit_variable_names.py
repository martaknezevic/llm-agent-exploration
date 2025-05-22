import os
from prompt import Prompt
from environments.abcEnv import ABCEnv
from environments.explicitEnv import ExplicitEnv
import hydra
from omegaconf import OmegaConf
from agent import OriginalAgent
from evaluation_explicit import EvaluatorExplicit as Evaluator
from utils import *
import json
import wandb
import re


def find_valid_steps(output: str):
    output_steps = []
    for line in output.split("\n"):
        if (
            line.startswith("Observation:")
            or line.startswith("Action:")
            or line.startswith("Thought:")
            or line.startswith("Hypothesis:")
        ) and line.strip() != "Observation:":
            output_steps.append(line)
        if line.startswith("Action"):
            break
    return "\n".join(output_steps)


def calculate_hyps(output: str):
    hyps = 0
    for line in output.split("\n"):
        if line.startswith("Hypothesis:"):
            hyps += 1
    return hyps


def is_thought_loop(agent_history, num_thoughts=5):
    seq_thoughts = 0
    for step in agent_history.split("\n")[::-1]:
        if step.startswith("Thought:"):
            seq_thoughts += 1
        else:
            seq_thoughts = 0
        if seq_thoughts >= num_thoughts:
            return True
    return False


def consolidate_set_functions(action_string: str) -> str:
    # Match all set function calls with key-value pairs
    pattern = r"set\((.*?)\)"

    # Find all the matches (key=value pairs inside set())
    matches = re.findall(pattern, action_string)

    # Extract all key-value pairs and join them into a single string
    arguments = ", ".join(matches)

    # Replace the original "set()" calls with a single one containing all arguments
    consolidated_action = f"Action: set({arguments})"

    return consolidated_action


def remove_assignment(input_str, key_to_remove):
    parts = input_str.strip()[12:-1].split(",")  # Remove "set(" and ")"
    filtered_parts = [
        part.strip()
        for part in parts
        if not part.strip().startswith(f"{key_to_remove}=")
    ]
    return f"set({', '.join(filtered_parts)})"


def parse_output_steps(output, dep_variable):
    steps = output.split("\n")
    res = []
    for step in steps:
        if step.startswith("set("):
            step = "Action: " + step
        if step.startswith("Action:"):
            if step.startswith("Action: set("):
                step = consolidate_set_functions(step)
                step = remove_assignment(step, dep_variable)
            match = re.search(r"(Action:.*\(.*?\))[^\)]*$", step)
            if match:
                step = match.group(1)
        res.append(step)
    return "\n".join(res)


def run_agent(config):
    if config.env.name == "abc":
        env = ABCEnv(config.env.equation)
    elif config.env.name == "explicit":
        env = ExplicitEnv(config.env.equation)
    else:
        raise ValueError("Environment name does not exist")

    env_variable_mapping = env.get_variable_mapping()
    config.env.variable_ranges = {
        env_variable_mapping.get(k, k): v for k, v in config.env.variable_ranges.items()
    }

    p = Prompt(config)
    p.build_initial_prompt()

    agent = OriginalAgent(
        model=config.agent.model,
        temperature=config.agent.temperature,
        seed=config.agent.seed,
    )
    num_actions = 0
    num_calls = 0
    num_hyps = 0
    explored_vars = ""
    agent_max_actions = config.agent.max_num_obs
    print(env.getProcessEquation())
    fin = False

    agent.reset_model()

    while num_actions <= agent_max_actions:
        prompt = p.get_prompt()
        prompt = prompt.replace("variable D", f"variable {env.output_variable}")
        if config.env.env_file.endswith("full.txt"):
            prompt = prompt.replace(
                f"that models variable {env.output_variable}",
                f"that models variable {env.output_variable} ({config.env.dep_var_name})",
            )

        if agent.history == "":
            print(prompt)

        if (
            num_actions == agent_max_actions
            or num_hyps >= 20
            or is_thought_loop(agent.history, num_thoughts=7)
            or num_calls >= 30
        ):
            fin = True
            if config.task.inquiry:
                prompt = prompt.replace(
                    f"Explore the variables using defined steps and find correct equation. After every new observation, use the Thought step to analyse it and reason about the observed change. Do not use multiple Thought steps in a sequence. First, collect a few observations and then propose hypothesis. Use the Hypothesis step explicitly to propose hypothesis. In order to confirm the hypothesis, first calculate the value of variable {env.output_variable} under current hypothesis step by step and then confirm and accept the hypothesis or discard it and propose new hypothesis. Do not write text outside of the defined steps.\nContinue the exploration process.",
                    "This is the exploration process:\n",
                )
            elif config.task.think:
                prompt = prompt.replace(
                    "Explore the variables using defined steps and find correct equation. After every new observation, use the Thought step to analyse it and reason about the observed change. Do not use multiple Thought steps in a sequence. Do not write text outside of the defined steps.\nContinue the exploration process.",
                    "This is the exploration process:\n",
                )
            else:
                prompt = prompt.replace(
                    "Explore the variables using defined steps and find correct equation. Do not write text outside of the defined steps.\nContinue the exploration process.",
                    "This is the exploration process:\n",
                )
            prompt += "\nAction: finish("
            prompt += f"\n\nWrite the finish Action. Do not write anything else, just write the final model of variable {env.output_variable} as argument of the finish Action. Do not write text outside of the defined steps."
            print(prompt)
        output = agent.act(
            prompt,
            stop=["\nObservation:", "\n\n"],
            do_sample=config.agent.do_sample,
            temperature=config.agent.temperature,
        )
        print(output)
        num_calls += 1
        output = output.replace("\_", "_")

        output = parse_output_steps(output, env.output_variable)
        try:
            if fin:
                action, output = parse_finish_action(output)
            else:
                if output.startswith("set("):
                    output = "Action: " + output
                action = output.split("Action:")[1].split("\n")[0].strip()
        except Exception as e:
            output_steps = find_valid_steps(output)
            num_hyps += calculate_hyps(output_steps)
            step = f"{output_steps}\n"
            agent.update(step)
            agent_history = agent.history
            p.build_prompt(agent_history)
            continue
        try:
            observation, done, model, var = env.step(action)

            if var:
                explored_vars += f"{var}\n"
            if not done:
                num_actions += 1

        except Exception as e:
            if "does not exist" in str(e):
                step = f"Error: {e}\n"
                done = False
                observation = None
            else:
                print(e)

        if done:
            agent.resulting_model = model
            output_steps = find_valid_steps(output)
            agent.update(f"{output_steps}\n")
            break

        json_str = ""
        if observation:
            # Convert the observation dictionary to a JSON string
            json_str = json.dumps(observation)

            # Replace all quotations in the keys
            json_str = json_str.replace('"', "")
            print("Observation: ", json_str)

        if json_str == "":
            json_str = step
        output_steps = find_valid_steps(output)
        num_hyps += calculate_hyps(output_steps)
        step = f"{output_steps}\nObservation: {json_str}\n"
        agent.update(step)
        agent_history = agent.history
        p.build_prompt(agent_history)
        print(step)

    evaluator = Evaluator(
        agent=agent,
        config=config,
    )

    compare_models, eval_fit = None, None
    try:
        compare_models = evaluator.compare_models(
            agent.resulting_model, agent.get_observations()
        )
        eval_fit = evaluator.evaluate_equation_fit(
            agent.resulting_model, agent.get_observations()
        )
    except Exception as e:
        print(e)

    with open(os.path.join(config.output_dir, "prompt.txt"), "w") as f:
        f.write(p.initial_prompt)
    with open(os.path.join(config.output_dir, "agent_history.txt"), "w") as f:
        f.write(agent.history)
    with open(os.path.join(config.output_dir, "results.json"), "w") as f:
        json.dump(
            {
                "compare_models": compare_models,
                "eval_fit": eval_fit,
                "agent_equation": agent.resulting_model,
            },
            f,
        )

    if config.wandb.use_wandb:
        wandb.log(
            {
                "num_actions": num_actions,
                "explored_vars": explored_vars,
                "agent_equation": agent.resulting_model,
                "correct_equation": env.getProcessEquation(),
                "eval_fit": eval_fit,
                "compare_models": compare_models,
            }
        )


@hydra.main(config_path="configs", config_name="explicit", version_base="1.1")
def main(config: OmegaConf):

    # seed everything
    set_seed(21)

    if config.wandb.use_wandb:
        # initialize wandb
        wandb.init(
            project=config.wandb.project_name,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            tags=config.wandb.tags,
        )

    run_agent(config)

    if config.wandb.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

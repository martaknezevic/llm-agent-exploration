import os
import openai
from prompt import Prompt
from environments.concentrationEnv import ConcentrationEnv
from environments.linearProcessEnv import LinearProcessEnv
from environments.abcEnv import ABCEnv
from environments.explicitEnv import ExplicitEnv
import hydra
from omegaconf import OmegaConf
from agent import OriginalAgent
from evaluation import Evaluator
from utils import *
import json
import wandb
import pickle
import re



def find_valid_steps(output: str):
    output_steps = []
    for line in output.split("\n"):
        if (
            line.startswith("Observation:")
            or line.startswith("Action:")
            or line.startswith("Thought:")
            or line.startswith("Memory:")
            or line.startswith("Hypothesis:")
        ) and line.strip() != 'Observation:':
            output_steps.append(line)
        if line.startswith("Action"):
            break
    return "\n".join(output_steps)

def calculate_hyps(output: str):
    hyps = 0
    for line in output.split('\n'):
        if line.startswith('Hypothesis:'):
            hyps += 1
    return hyps

def is_thought_loop(agent_history, num_thoughts=5):
    seq_thoughts = 0
    for step in agent_history.split('\n')[::-1]:
        if step.startswith('Thought:'):
            seq_thoughts += 1
        else:
            seq_thoughts = 0
        if seq_thoughts >= num_thoughts:
            return True
    return False

def parse_output_steps(output):
    steps = output.split('\n')
    res = []
    for step in steps:
        if step.startswith('set('):
            step = 'Action: ' + step
        if step.startswith('finish('):
            step = 'Action: ' + step
        if step.startswith('Action:'):
            match = re.search(r"(Action:.*\(.*?\))[^\)]*$", step)
            if match:
                step = match.group(1)
        res.append(step)
    return '\n'.join(res)

def check_and_verify_hyps(output, evaluator, agent):
    verified = False
    a_b4_h = False
    correct_hypothesis = False
    
    output_steps = []
    for line in output.split('\n'):
        output_steps.append(line)
        if line.startswith('Action:'):
            a_b4_h = True
        elif line.startswith('Hypothesis:'):
            hypothesis_step_content = line[len('Hypothesis:'):].strip()
            if 'D=' in hypothesis_step_content:
                hypothesis = hypothesis_step_content[hypothesis_step_content.find('D='):]
            elif 'D =' in hypothesis_step_content:
                hypothesis = hypothesis_step_content[hypothesis_step_content.find('D ='):]
            else:
                hypothesis = hypothesis_step_content


            try:
                compare_models = evaluator.compare_models(
                    hypothesis, agent.get_observations()
                )
                print(f'compare models:{compare_models}')
                eval_fit, conclusion = evaluator.conclude(
                    hypothesis, agent.get_observations()
                )
                print(f'eval_fit:{eval_fit}')
                print(f'conclusion:{conclusion}')

                if eval_fit and not compare_models:
                    conclusion = "Although all the previous observations fit hypothesis, I will explore more because hypothesis might not be correct.\n"    

                elif eval_fit and compare_models:
                    correct_hypothesis = True

            except Exception as e:
                print(e)
                conclusion = 'This is wrong hypothesis format. The hypothesis should be written as inline expression with operators in plain text using only following operators: +, -, *, / and (), and existing variables.\n'

            output_steps.append(f'Thought: {conclusion}')
            verified = True
            break
        
    output = '\n'.join(output_steps)   
    return output, verified, a_b4_h, correct_hypothesis

def run_agent(config):
    if config.env.name == "linear":
        env = LinearProcessEnv(float(config.env.a), float(config.env.b))
    elif config.env.name == "concentration":
        env = ConcentrationEnv()
    elif config.env.name == "abc":
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
        config=config
    )
    
    evaluator = Evaluator(
            agent=agent,
            config=config,
        )
    
    num_actions = 0
    num_calls = 0
    num_hyps =0
    explored_vars = ""
    agent_max_actions = config.agent.max_num_obs
    print(env.getProcessEquation())
    fin=False
    end = False
    
    #agent.reset_model()
    
    while num_actions <= agent_max_actions:
        prompt = p.get_prompt()
        
        if agent.history == '':
            prompt = prompt.replace('Continue the exploration process.', 'Start the exploration process.')
        
        if num_actions == agent_max_actions or num_hyps>=20 or is_thought_loop(agent.history, num_thoughts=7) or num_calls > config.agent.max_calls or end == True:
            # prompt += 'Action: finish'
            fin = True
            if config.task.inquiry:
                if config.task.zero_shot:
                    prompt = prompt.replace('Explore the variables using defined steps and find correct equation. After every new observation, use the Thought step to analyse it and reason about the observed change. Do not use multiple Thought steps in a sequence. First, collect a few observations and then propose hypothesis. Use the Hypothesis step explicitly to propose hypothesis. Write the hypothesis as a mathematical equation only without any additional text. In order to confirm the hypothesis, first calculate the value of D under current hypothesis step by step and then confirm and accept the hypothesis or discard it and propose new hypothesis. Do not write text outside of the defined steps.\nContinue the exploration process.', 'This is the exploration process:\n')
                else:
                    prompt = prompt.replace('Explore the variables using defined steps and find correct equation. After every new observation, use the Thought step to analyse it and reason about the observed change. Do not use multiple Thought steps in a sequence. First, collect a few observations and then propose hypothesis. Use the Hypothesis step explicitly to propose hypothesis. In order to confirm the hypothesis, first calculate the value of D under current hypothesis step by step and then confirm and accept the hypothesis or discard it and propose new hypothesis. Do not write text outside of the defined steps.\nContinue the exploration process.', 'This is the exploration process:\n')
            elif config.task.think:
                prompt = prompt.replace('Explore the variables using defined steps and find correct equation. After every new observation, use the Thought step to analyse it and reason about the observed change. Do not use multiple Thought steps in a sequence. Do not write text outside of the defined steps.\nContinue the exploration process.', 'This is the exploration process:\n')
            else:
                prompt = prompt.replace('Explore the variables using defined steps and find correct equation. Do not write text outside of the defined steps.\nContinue the exploration process.', 'This is the exploration process:\n')           
            #prompt += '\n\nFinish the exploration by writing the Action finish. Do not continue exploring the variables. Do not write text outside of the defined steps. Finish the exploration and write the model of variable D as the argument of the finish Action.\n'
            prompt += '\nAction: finish('
            prompt += '\n\nWrite the finish Action. Do not write anything else, just write the final model of variable D as argument of the finish Action. Do not write text outside of the defined steps.'
            print(prompt)
        output = agent.act_old(prompt, stop=["\nObservation:", '\n\n'], do_sample=config.agent.do_sample, temperature=config.agent.temperature)
        output = parse_output_steps(output)
        num_calls += 1
        
        if config.task.inquiry and config.task.verify:
            output, verified, a_b4_h, correct_hypothesis = check_and_verify_hyps(output, evaluator, agent)
            if not a_b4_h and verified:
                output_steps = find_valid_steps(output)
                num_hyps += calculate_hyps(output_steps)
                step = f"{output_steps}\n"
                agent.update(step)
                agent_history = agent.history
                p.build_prompt(agent_history)
                
                if correct_hypothesis:
                    end = True
                continue                  
                 
        try:
            if fin:
                action, output = parse_finish_action(output)
            else:
                if output.startswith('set('):
                    output = 'Action: ' + output
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
            if 'does not exist' in str(e):
                json_str = f"Error: {e}\n"
                done = False
                observation = None
            else:
                print(e)

        if done:
            agent.resulting_model = model
            output_steps = find_valid_steps(output)
            agent.update(f"{output_steps}\n")
            break

        if observation:
            # Convert the observation dictionary to a JSON string
            json_str = json.dumps(observation)

            # Replace all quotations in the keys
            json_str = json_str.replace('"', "")
            print('Observation: ', json_str)


        output_steps = find_valid_steps(output)
        num_hyps += calculate_hyps(output_steps)
        step = f"{output_steps}\nObservation: {json_str}\n"
        agent.update(step)
        agent_history = agent.history
        p.build_prompt(agent_history)


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
        json.dump({'compare_models': compare_models, 'eval_fit': eval_fit, 'agent_equation': agent.resulting_model}, f)

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



@hydra.main(config_path="configs", config_name="base", version_base="1.1")
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

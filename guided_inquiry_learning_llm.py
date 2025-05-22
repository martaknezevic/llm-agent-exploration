import os
from prompt import Prompt
from environments.abcEnv import ABCEnv
import hydra
from omegaconf import OmegaConf
from agent import OriginalAgent
from evaluation import Evaluator
from utils import *
import json
import wandb
from dataclasses import dataclass

@dataclass
class ExplorationStatus:
    num_actions: int = 0
    num_hyps: int = 0
    explored_vars: str = ''
    fin: bool = False
    num_loop: int = 0
    next_step: str = None
    
    def __post_init__(self):
        if self.next_step not in ['action', 'observation', 'hypothesis', 'thought']:
            raise ValueError(f'Invalid next step: {self.next_step}')


def find_valid_steps(output: str):
    output_steps = []
    for line in output.split("\n"):
        if (
            line.startswith("Observation:")
            or line.startswith("Action:")
            or line.startswith("Thought:")
            or line.startswith("Hypothesis:")
        ) and line.strip() != 'Observation:':
            output_steps.append(line)
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

def add_instruction(prompt, next_step, instructions, config):
    if next_step == 'thought' and config.task.guided_instructions:
        next_step = 'thought_guided_last'
    elif next_step == 'thought' and config.task.guided_reasoning:
        next_step = 'thought_only_guided'
    
        
    if next_step == 'action' and config.task.guided_instructions:
        next_step = 'action_guided'
    elif next_step == 'action' and config.task.guided_strategy:
        next_step = 'action_guided'
        
    if next_step == 'hypothesis' and config.task.zero_shot:
        next_step = 'hypothesis_zero'
 
        
    instruction = instructions[next_step]
    return f'{prompt}\n\n{instruction}\n'

def parse_act(output, agent, env, evaluator):
    interact = False
    hyp = False
    compare_models = False
    finish=False

    if output.startswith('finish'):
        output = 'Action: ' + output
    
    if output.startswith('Thought:') and output[-1] != '\n':
        output = output + '\n'
    
    if output.startswith('Plan:') or output.startswith('Thought:'):
        step = output
    elif output.startswith('Action'):
        action = output.split("Action:")[1].split("\n")[0].strip()
        #print(output)
        try:
            observation, done, model, var = env.step(action)
        except Exception as e:
            if 'does not exist' in str(e):
                step = f"{output}\nError: {e}\n"
                done = False
                observation = None
                
        if done:
            agent.resulting_model = model
            finish = True
            
            hyp=model
            try:
                compare_models = evaluator.compare_models(
                model, agent.get_observations()
            )
            except Exception as e:
                print(e)
            step = f"{output}"
        
        if observation:
            # Convert the observation dictionary to a JSON string
            json_str = json.dumps(observation)

            # Replace all quotations in the keys
            json_str = json_str.replace('"', "")

            #print('Observation: ', json_str)
            step = f"{output.strip()}\nObservation: {json_str}\n"

        interact = True
        
    elif output.startswith('Hypothesis:'):
        hypothesis_step_content = output[len('Hypothesis:'):].strip()
        hyp = hypothesis_step_content
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
            
            
        except Exception as e:
            print(e)
            conclusion = 'This is wrong hypothesis format. The hypothesis should be written as inline expression with operators in plain text using only following operators: +, -, *, / and (), and existing variables.\n'
            
        step = f'{output.strip()}\nThought: {conclusion}'     

    print(step)
    return interact, hyp, compare_models, step, finish
    
def validate_output(output, next_step):
    valid_steps = ['Action:', 'Thought:', 'Hypothesis:']
    for step in valid_steps:
        if output.startswith(step):
            return output
    
    return f'{next_step.capitalize()}: {output.strip()}\n'

def run_agent(config):
    
    assert not (config.task.guided_instructions and config.task.guided_strategy), "You can not use both guided instructions and guided strategy at the same time."
    assert not (config.task.guided_instructions and config.task.guided_reasoning), "You can not use both guided instructions and guided reasoning at the same time."
    assert not (config.task.guided_strategy and config.task.guided_reasoning), "You can not use both guided strategy and guided reasoning at the same time."
    
    if config.env.name == "abc":
        env = ABCEnv(config.env.equation)
    else:
        raise ValueError(f"Environment name {config.env.equation} does not exist")

    env_variable_mapping = env.get_variable_mapping()
    config.env.variable_ranges = {
        env_variable_mapping.get(k, k): v for k, v in config.env.variable_ranges.items()
    }

    p = Prompt(config)
    with open('./prompts/guided_instructions.json', 'r') as f:
        instructions = json.load(f)
        
    p.build_initial_prompt()

    agent = OriginalAgent(
        model=config.agent.model,
        temperature=config.agent.temperature,
        seed=config.agent.seed,
    )
    
    print(env.getProcessEquation())
    
    if not config.agent.model.startswith("gpt"):    
        agent.reset_model()
        
    evaluator = Evaluator(
            agent=agent,
            config=config,
        )
    print(p.get_prompt())
    
    status = ExplorationStatus(next_step='action')
    la1_steps = config.agent.la1_steps      
    
    # AOT - initial exploration
    while status.num_actions <= la1_steps + 1:
        prompt = p.get_prompt()
        prompt = add_instruction(prompt, status.next_step, instructions, config)
        
        output = agent.act(prompt, stop=['\n'], max_new_tokens=200, do_sample=config.agent.do_sample, temperature=config.agent.temperature)
        if output == '' or output == '\n':
            status.num_loop += 1
        output = validate_output(output, status.next_step)
        
        interact, hyp, compare_models, step, finish = parse_act(output, agent, env, evaluator)  
        
        if interact:
            if finish:
                break
            status.num_actions += 1
            status.next_step = 'thought'
            
        elif status.next_step == 'thought':
            if 'Action:' in step:
                step = step[:step.index('Action:')].strip()

            status.next_step = 'action'
        
        if status.num_actions > la1_steps:
            status.num_actions -= 1
            break
        
        agent.update(step)
        agent_history = agent.history
        p.build_prompt(agent_history)
        
        if status.num_actions == la1_steps and status.next_step != 'thought':
            break
        
        
        
    # iterative loops of inquiry
    if not finish: 
        status.next_step = 'hypothesis'    
        for ee in range(config.agent.exploration_epochs):
            num_seq_hyp = 0
            while status.num_actions <= config.agent.la1_steps + config.agent.la2_steps * (ee + 1) + 1:
                prompt = p.get_prompt()
                prompt = add_instruction(prompt, status.next_step, instructions, config)

                if status.next_step == 'thought':
                    prompt += "If previous hypothesis have been discarded, try to observe correct pattern and find correct model for variable D.\n "
                if status.next_step == 'hypothesis' and num_seq_hyp > 0:
                    prompt += "Write the hypothesis in the correct format.\n "
                print(prompt)
                output = agent.act(prompt, stop=['\n'], max_new_tokens=200, do_sample=config.agent.do_sample, temperature=config.agent.temperature)
                if output == '' or output == '\n':
                    status.loop += 1
                interact, hyp, compare_models, step, finish = parse_act(output, agent, env, evaluator)
                
                if status.next_step == 'hypothesis' and 'This is wrong hypothesis format.' in step and num_seq_hyp < 3:
                    agent.update(step)
                    agent_history = agent.history
                    p.build_prompt(agent_history)
                    num_seq_hyp += 1
                    continue

                if hyp and not finish:
                    status.next_step = 'action'

                elif interact:
                    if finish:
                        break

                    status.next_step = 'thought'
                    status.num_actions += 1

                else:
                    status.next_step = 'action'    
                    
                if config.agent.la2_steps == 0:
                    agent.update(step)
                    agent_history = agent.history
                    p.build_prompt(agent_history)

                
                
                agent.update(step)
                agent_history = agent.history
                p.build_prompt(agent_history)
                
                if status.num_actions == config.agent.la1_steps + config.agent.la2_steps * (ee + 1) and status.next_step != 'thought':
                    break

                if compare_models:
                    break
            if compare_models or finish:
                break
            
            status.next_step = 'hypothesis'
        
        if not finish:
            # finish exploration
            prompt = p.get_prompt()
            prompt = add_instruction(prompt, 'finish', instructions, config)
            output = agent.act(prompt, stop=['\n'], max_new_tokens=200, do_sample=config.agent.do_sample, temperature=config.agent.temperature)

            action, output = parse_finish_action(output)
            agent.update(output)
            p.build_prompt(agent.history)

            print(f'Last finish action: {action}')
            _, done, model, _ = env.step(action)
            agent.resulting_model = model
            
            try:
                compare_models = evaluator.compare_models(
                    agent.resulting_model, agent.get_observations()
                )
                print(f'compare models:{compare_models}')
            except Exception as e:
                print(e)
    else:
        agent.update(step)
        agent_history = agent.history
        p.build_prompt(agent_history)
    
    ####################################################################################################################
    
    with open(os.path.join(config.output_dir, "prompt.txt"), "w") as f:
        f.write(p.initial_prompt)
    with open(os.path.join(config.output_dir, "final_prompt.txt"), 'w') as f:
        f.write(p.get_prompt())
    with open(os.path.join(config.output_dir, "agent_history.txt"), "w") as f:
        f.write(agent.history)
    with open(os.path.join(config.output_dir, "results.json"), "w") as f:
        json.dump({'compare_models': compare_models, 'agent_equation': agent.resulting_model}, f)

    if config.wandb.use_wandb:
        wandb.log(
            {
                "num_actions": status.num_actions,
                "explored_vars": status.explored_vars,
                "agent_equation": agent.resulting_model,
                "correct_equation": env.getProcessEquation(),
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

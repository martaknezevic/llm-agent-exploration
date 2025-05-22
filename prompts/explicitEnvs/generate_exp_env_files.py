import json
import os

size = {
    2: 'two',
    3: 'three',
    4: 'four',
}

with open('/code/knezevic/llm-simulations/prompts/explicitEnvs/explicit_template.txt', 'r') as f:
    template = f.read()

with open('/code/knezevic/llm-simulations/prompts/explicitEnvs/explicit_new.json', 'r') as f:
    data = json.load(f)
    
    for d in data:
        d_template = template
        input_vars = d['input_variables']
        output_var = d['output_variable']
        
        path = d['file_path']
        
        dir_name = os.path.dirname(path)
        file_name = os.path.basename(path).split('.')[0]
        
        available_vars = ', '.join([var['acronym'] for var in input_vars[:-1]]) + ' and ' + input_vars[-1]['acronym']
        dep_var = f'variable {output_var["acronym"]}'
        using_vars = f'variables {available_vars}'
        
        with open(os.path.join(f"/code/knezevic/llm-simulations/{dir_name}", f'{file_name}_acr.txt'), 'w') as f:
            tmp = d_template.format(
                size[len(input_vars)],
                available_vars,
                dep_var,
                using_vars
            )
            f.write(tmp)
            
        available_vars_full = ', '.join([f"{var['acronym']} ({var['name']})" for var in input_vars[:-1]]) + ' and ' + f"{input_vars[-1]['acronym']} ({input_vars[-1]['name']})"
        dep_var_full = f"variable {output_var['acronym']} ({output_var['name']})"
        using_vars = f'variables {available_vars}'
        
        with open(os.path.join(f"/code/knezevic/llm-simulations/{dir_name}", f'{file_name}_full.txt'), 'w') as f:
            tmp = d_template.format(
                size[len(input_vars)],
                available_vars_full,
                dep_var_full,
                using_vars
            )
            f.write(tmp)
        
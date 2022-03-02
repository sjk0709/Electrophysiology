import sys, os
import copy
import time
import pickle

sys.path.append('../Lib')
import ga_configs
import ga_vc_optimization

def create_folder(folder_path):
    '''Create folder'''
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print('"%s" has been created.'%(folder_path))
        else:
            print("The folder already exists.")
    except OSError:
        print('Error: create_folder(). : ' + folder_path)
        
        
WITH_ARTEFACT = True

VCO_CONFIG = ga_configs.VoltageOptimizationConfig(
    window=2,
    step_size=2,
    steps_in_protocol=4,
    step_duration_bounds=(5, 1000),
    step_voltage_bounds=(-120, 60),
    target_current='',
    population_size=2,
    max_generations=2,
    mate_probability=0.9,
    mutate_probability=0.9,
    gene_swap_probability=0.2,
    gene_mutation_probability=0.1,
    tournament_size=2,
    step_types=['step', 'ramp'],
    with_artefact=WITH_ARTEFACT,
    model_name='BR1977')


LIST_OF_CURRENTS = ['I_Na', 'I_si', 'I_K1', 'I_x1']# ['I_Na', 'I_Kr', 'I_Ks', 'I_To', 'I_F', 'I_CaL', 'I_K1']

def main():
    """Run parameter tuning or voltage clamp protocol experiments here
    """
    create_folder('./ga_results')
        
    vco_dir_name = f'trial_steps_ramps_{VCO_CONFIG.model_name}_{VCO_CONFIG.population_size}_{VCO_CONFIG.max_generations}_{VCO_CONFIG.steps_in_protocol}_{VCO_CONFIG.step_voltage_bounds[0]}_{VCO_CONFIG.step_voltage_bounds[1]}'

    if not vco_dir_name in os.listdir('ga_results'):
        os.mkdir(f'ga_results/{vco_dir_name}')

    for c in LIST_OF_CURRENTS:
        f = f"./ga_results/{vco_dir_name}/ga_results_{c}_artefact_{WITH_ARTEFACT}"
        print(f"Finding best protocol for {c}. Writing protocol to: {f}")
        VCO_CONFIG.target_current = c
        result = ga_vc_optimization.start_ga(VCO_CONFIG)
        print("="*100)
        pickle.dump(result, open(f, 'wb'))
        
        
        
if __name__ == '__main__':
    main()
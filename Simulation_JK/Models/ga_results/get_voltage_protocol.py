import pickle
from os import listdir
import matplotlib.pyplot as plt
import mod_protocols as protocols
# import mod_kernik as kernik
import bisect

def plot_shorten_with_vcmd(trial_conditions, model_name, is_vc_only=False):
    path_to_data = f"{trial_conditions}"

    files = listdir(path_to_data)

    for f in files:
        if ('shorten' in f) and ('pkl' in f):
            file_name = f
    print(f"{path_to_data}/{file_name}")
    short_protocol = pickle.load(open(f"{path_to_data}/{file_name}", 'rb'))
    print(short_protocol.steps)  
    print(short_protocol.get_voltage_change_endpoints())
    print(short_protocol.get_voltage_change_startpoints())      
    print("="*50)    
    start_point_li = short_protocol.get_voltage_change_startpoints()
    end_point_li = short_protocol.get_voltage_change_endpoints()
    for step_index, current_step in enumerate(short_protocol.steps):        
        if isinstance(current_step, protocols.VoltageClampStep):
            # print("voltage :", current_step.voltage)
            # print("duration :", current_step.duration)
            # return current_step.voltage
            # print(current_step.duration)
            continue
        elif isinstance(current_step, protocols.VoltageClampRamp):            
            print("start :", start_point_li[step_index])
            print("end :", end_point_li[step_index])
            print("voltage_start :", current_step.voltage_start)
            print("voltage_end :", current_step.voltage_end)            
            print("duration :", current_step.duration)        
            a, b = get_voltage_clamp_step_tangent_yIntercept(start_point_li[step_index], current_step.voltage_start, end_point_li[step_index], current_step.voltage_end) 
            print("Tangent :", a)
            print("y-intercept", b)
            # return current_step.get_voltage(time_into_step)
            # get_voltage_clamp_step_range_tangent()       
            print("-"*50)     
        else:
            # return current_step.get_voltage(time_into_step)
            # print(current_step.get_voltage(time_into_step))
            continue    
    print("="*50)
    print(f'The protocol is {short_protocol.get_voltage_change_endpoints()[-1]} ms')
    

    if is_vc_only:
        short_protocol.get_voltage_clamp_protocol()
        return

    # time = 1000
    # step_index = bisect.bisect_left(
    #     short_protocol.get_voltage_change_endpoints(),
    #     time)
    # print(step_index)
    # current_step = short_protocol.steps[step_index]
    # time_into_step = time - short_protocol.get_voltage_change_startpoints()[
    #             step_index]        
    # if isinstance(current_step, protocols.VoltageClampStep):
    #     # print("voltage :", current_step.voltage)
    #     # print("duration :", current_step.duration)
    #     return current_step.voltage
    # elif isinstance(current_step, protocols.VoltageClampRamp):
    #     # print(time_into_step)
    #     return current_step.get_voltage(time_into_step)
    # else:
    #     return current_step.get_voltage(time_into_step)


def get_voltage_clamp_step_tangent_yIntercept(t1, V1, t2, V2):
    '''
    y=ax+b
    y-V1 = (V2-V1)/(t2-t1)*(x-t1)  <-  (t1, V1) , (t2, V2)
    return a, b
    '''
    a = (V2 - V1)/(t2 - t1)
    b = -a*t1 + V1
    return a, b
    



def main():
    trial_conditions = "trial_steps_ramps_Kernik_200_50_4_-120_60"
    model_name = 'Kernik'

    plot_shorten_with_vcmd(trial_conditions, model_name, is_vc_only=True)


if __name__ == '__main__':
    main()

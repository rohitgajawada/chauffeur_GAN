"""
    A simple factory module that returns instances of possible modules 

"""

from .models import CoILICRA, coil_ganmodules_task_wdgrl, unit_networks


def CoILModel(architecture_name):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if architecture_name == 'wgangp_lsd':

        return coil_ganmodules_task_wdgrl._netF()
    
    elif architecture_name == 'coil_icra':

        return CoILICRA()
    
    elif architecture_name == 'coil_unit':
        
        return unit_networks._netF(), unit_networks.VAEGen()

    else:

        raise ValueError(" Not found architecture name")

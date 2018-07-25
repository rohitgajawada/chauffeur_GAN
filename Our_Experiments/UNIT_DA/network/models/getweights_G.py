import torch
from collections import OrderedDict
import coil_ganmodules_task
# import coil_ganmodules_nopatch

import coil_icra

# modelG = coil_ganmodules_nopatch._netG()
# modelG_new = coil_ganmodules_task._netG()

modelF_IL = torch.load('F_IL_edited.wts')
modelF = coil_ganmodules_task._netF()

### Edit IL model to remove running_mean

# F_IL_edited = OrderedDict()
# for key in modelF_IL.keys():
#     if 'running' in key:
#         pass
#     else:
#         F_IL_edited[key] = modelF_IL[key]

# torch.save(F_IL_edited, 'F_IL_edited.wts')

F_stdict = modelF.state_dict()

### IL weights to netF ###

for i, keys in enumerate(zip(F_stdict.keys(), modelF_IL.keys())):
    key1, key2 = keys
    print (key1, key2)
    F_stdict[key1] = modelF_IL[key2]
    if i == 13:
        break

torch.save(F_stdict, 'netF_IL_Pretrained.wts')



##########################

# # Take p1, p2, p3 from G
# ckpt = torch.load('best_modelG.pth')
# print(ckpt.keys())


# modelG.load_state_dict(ckpt['stateG_dict'])

# # Take decoder part for verification
# F_stdict = modelF.state_dict()
# G_stdict = modelG.state_dict()
# G_newstdict = modelG_new.state_dict()

# for i, key in enumerate(F_stdict.keys()):
#     F_stdict[key] = G_stdict[key]
#     if i == 13:
#         break

# for i, key in enumerate(G_stdict.keys()):
#     if i > 13:
#         G_newstdict[key] = G_stdict[key]

# torch.save(F_stdict, 'netF_GAN_Pretrained.wts')
# torch.save(G_newstdict, 'netG_GAN_Pretrained.wts')


### Testing loading the wts

# modelF.load_state_dict(torch.load('netF_GAN_Pretrained.wts'))
# modelG_new.load_state_dict(torch.load('netG_GAN_Pretrained.wts'))

# for i, key in enumerate(modelG_new.keys()):
#     print(i, key)


# netF = coil_ganmodules_task._netF()
# netG = coil_ganmodules_task._netG()

# netF =
# netG =

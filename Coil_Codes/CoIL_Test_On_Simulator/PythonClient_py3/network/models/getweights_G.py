import torch
# import coil_ganmodules_nopatch
import coil_ganmodules_task


# modelG = coil_ganmodules_nopatch._netG()
modelF = coil_ganmodules_task._netF()
modelG_new = coil_ganmodules_task._netG()

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

modelF.load_state_dict(torch.load('netF_GAN_Pretrained.wts'))
modelG_new.load_state_dict(torch.load('netG_GAN_Pretrained.wts'))

for i, key in enumerate(modelG_new.keys()):
    print(i, key)


# netF = coil_ganmodules_task._netF()
# netG = coil_ganmodules_task._netG()

# netF =
# netG =

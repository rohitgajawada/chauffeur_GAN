import torch
import coil_ganmodules_task
import coil_icra


model_IL = torch.load('best_loss_20-06_EpicClearWeather.pth')
model_IL_state_dict = model_IL['state_dict']

modelF = coil_ganmodules_task._netF()
modelF_state_dict = modelF.state_dict()

print(len(modelF_state_dict.keys()), len(model_IL_state_dict.keys()))

for i, keys in enumerate(zip(modelF_state_dict.keys(), model_IL_state_dict.keys())):
    newkey, oldkey = keys
    if newkey.split('.')[0] == "branch" and oldkey.split('.')[0] == "branches":
        print("No Transfer of ",  newkey, " to ", oldkey)
    else:
        print("Transferring ", newkey, " to ", oldkey)
        modelF_state_dict[newkey] = model_IL_state_dict[oldkey]

modelF.load_state_dict(modelF_state_dict)
print("IL Model Loaded!")

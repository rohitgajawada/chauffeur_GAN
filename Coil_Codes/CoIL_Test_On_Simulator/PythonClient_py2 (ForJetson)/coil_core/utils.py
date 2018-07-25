import torch

# def adjust_learning_rate(opt, optimizer, epoch):
#     """
#     Adjusts the learning rate every epoch based on the selected schedule
#     """
#     epoch = copy.deepcopy(epoch)
#     lr = opt.maxlr
#     wd = opt.weightDecay
#     if opt.learningratescheduler == 'imagenetscheduler':
#         if epoch >= 1 and epoch <= 18:
#             lr = 1e-3
#             wd = 5e-5
#         elif epoch >= 19 and epoch <= 29:
#             lr = 5e-4
#             wd = 5e-5
#         elif epoch >= 30 and epoch <= 43:
#             lr = 1e-4
#             wd = 0
#         elif epoch >= 44 and epoch <= 52:
#             lr = 5e-5
#             wd = 0
#         elif epoch >= 53:
#             lr = 2e-5
#             wd = 0
#         if opt.optimType=='sgd':
#             lr *= 10
#         opt.lr = lr
#         opt.weightDecay = wd
#     if opt.learningratescheduler == 'decayscheduler':
#         while epoch >= opt.decayinterval:
#             lr = lr/opt.decaylevel
#             epoch = epoch - opt.decayinterval
#         lr = max(lr,opt.minlr)
#         opt.lr = lr
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#         param_group['weight_decay'] = wd

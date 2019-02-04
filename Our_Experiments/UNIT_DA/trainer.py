"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen, _netF, _netF_CoordConv, _netF_Wayve
from utilsUNIT import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

from input import CoILDataset, BatchSequenceSampler, splitter
from network.loss_task import TaskLoss
from torchvision import transforms

import math

class UNIT_Trainer(nn.Module):
	def __init__(self, hyperparameters, coordconv=False, no_speed=False):
		super(UNIT_Trainer, self).__init__()
		lr = hyperparameters['lr']
		self.no_speed = no_speed
		# task part
		if coordconv:
			self.netF = _netF_Wayve().cuda()
		else:
			self.netF = _netF().cuda()
		# netF.train()
		self.Task_Loss = TaskLoss()
		self.mseloss = nn.MSELoss()

		# Initiate the networks
		self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
		self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
		self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
		self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
		self.instancenorm = nn.InstanceNorm2d(512, affine=False)

		# Setup the optimizers
		beta1 = hyperparameters['beta1']
		beta2 = hyperparameters['beta2']
		dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
		gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters()) + list(self.netF.parameters())
		self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
										lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
		self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
										lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
		self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
		self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

		# Network weight initialization
		self.apply(weights_init(hyperparameters['init']))
		self.dis_a.apply(weights_init('gaussian'))
		self.dis_b.apply(weights_init('gaussian'))

		# Load VGG model if needed
		if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
			self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
			self.vgg.eval()
			for param in self.vgg.parameters():
				param.requires_grad = False

	def recon_criterion(self, input, target):
		return torch.mean(torch.abs(input - target))

	def forward(self, x_a, x_b):
		self.eval()
		x_a.requires_grad=False
		x_b.requires_grad=False
		h_a, _ = self.gen_a.encode(x_a)
		h_b, _ = self.gen_b.encode(x_b)
		x_ba = self.gen_a.decode(h_b)
		x_ab = self.gen_b.decode(h_a)
		self.train()
		return x_ab, x_ba

	def __compute_kl(self, mu):
		# def _compute_kl(self, mu, sd):
		# mu_2 = torch.pow(mu, 2)
		# sd_2 = torch.pow(sd, 2)
		# encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
		# return encoding_loss
		mu_2 = torch.pow(mu, 2)
		encoding_loss = torch.mean(mu_2)
		return encoding_loss

	def gen_update(self, x_a, x_b, float_data, hyperparameters):
		self.gen_opt.zero_grad()

		# init data
		full_dataset = hyperparameters['train_dataset_name']
		real_dataset = hyperparameters['target_domain_path']
		dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

		# encode
		h_a, n_a = self.gen_a.encode(x_a)
		h_b, n_b = self.gen_b.encode(x_b)
		# decode (within domain)
		x_a_recon = self.gen_a.decode(h_a + n_a)
		x_b_recon = self.gen_b.decode(h_b + n_b)
		# decode (cross domain)
		x_ba = self.gen_a.decode(h_b + n_b)
		x_ab = self.gen_b.decode(h_a + n_a)
		# encode again
		h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
		h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
		# decode again (if needed)
		x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
		x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

		# #task part
		identity_embed = h_a
		cycle_embed = h_a_recon

		# Steering Augmentation
		###########################################
		camera_angle = float_data[:,26,:]
		camera_angle = camera_angle.cuda()#self._config.variable_names.index('Angle'),i]
		# print ("Camera angle", camera_angle[0])
		steer = float_data[:,0,:]
		# print("Original", steer[0])
		steer = steer.cuda()
		speed = float_data[:,10,:]
		speed = speed.cuda()
		# print (steer)

		time_use =  1.0
		car_length = 3.0
		extra_factor = 2.5
		threshold = 1.0

		pos = camera_angle > 0.0
		pos = pos.type(torch.FloatTensor)
		neg = camera_angle <= 0.0
		neg = neg.type(torch.FloatTensor)
		pos = pos.cuda()
		neg = neg.cuda()

		rad_camera_angle = math.pi*(torch.abs(camera_angle)) / 180.0
		val = extra_factor *(torch.atan((rad_camera_angle*car_length)/(time_use*speed+0.05)))/3.1415
		# print(val)
		steer -= pos * torch.min(val , torch.Tensor([0.6]).cuda())
		steer += neg * torch.min(val, torch.Tensor([0.6]).cuda())

		# print("val", val[0])
		# print ("speed", speed[0])

		steer = steer.cpu()
		float_data[:,0,:] = steer
		float_data[:, 0, :][float_data[:,0,:] > 1.0] = 1.0
		float_data[:, 0, :][float_data[:,0,:] < -1.0] = -1.0
		###########################################

		identity_task = self.netF(identity_embed, Variable(dataset.extract_inputs(float_data)).cuda())
		cycle_task = self.netF(cycle_embed, Variable(dataset.extract_inputs(float_data)).cuda())
		controls = Variable(float_data[:, dataset.controls_position(), :])

		# task loss
		if self.no_speed:
			self.lossF_identity_task = self.mseloss(identity_task, Variable(dataset.extract_targets(float_data)).cuda())
			self.lossF_cycle_task = self.mseloss(cycle_task, Variable(dataset.extract_targets(float_data)).cuda())
		else:
			print("Fast and the furious")
			self.lossF_identity_task = self.Task_Loss.MSELoss(identity_task, Variable(dataset.extract_targets(float_data)).cuda(),
										 controls.cuda(), Variable(dataset.extract_inputs(float_data)).cuda())
			self.lossF_cycle_task = self.Task_Loss.MSELoss(cycle_task, Variable(dataset.extract_targets(float_data)).cuda(),
										 controls.cuda(), Variable(dataset.extract_inputs(float_data)).cuda())

		self.lossF_task = self.lossF_identity_task + self.lossF_cycle_task

		# reconstruction loss
		# print(x_a_recon[0][0][:5][:5])
		# print("Help loss:", self.recon_criterion(x_a_recon, x_a))
		# print("identity task", identity_task[0])
		# print("cycle task", cycle_task[0])

		self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
		self.loss_gen_latrecon_x_a = self.recon_criterion(h_a_recon, h_a)

		self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
		self.loss_gen_latrecon_x_b = self.recon_criterion(h_b_recon, h_b)

		self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
		self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
		self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
		self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
		self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
		self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
		# GAN loss
		self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
		self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
		# domain-invariant perceptual loss
		self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
		self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
		# total loss
		self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
							  hyperparameters['gan_w'] * self.loss_gen_adv_b + \
							  hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
							  hyperparameters['recon_x_w'] * self.loss_gen_latrecon_x_a + \
							  hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
							  hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
							  hyperparameters['recon_x_w'] * self.loss_gen_latrecon_x_b + \
							  hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
							  hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
							  hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
							  hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
							  hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
							  hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
							  hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
							  hyperparameters['task_w'] * self.lossF_task
		self.loss_gen_total.backward()
		self.gen_opt.step()

	def compute_vgg_loss(self, vgg, img, target):
		img_vgg = vgg_preprocess(img)
		target_vgg = vgg_preprocess(target)
		img_fea = vgg(img_vgg)
		target_fea = vgg(target_vgg)
		return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

	def sample(self, x_a, x_b):
		self.eval()
		x_a.requires_grad = False
		x_b.requires_grad=False
		x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
		for i in range(x_a.size(0)):
			h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
			h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
			x_a_recon.append(self.gen_a.decode(h_a))
			x_b_recon.append(self.gen_b.decode(h_b))
			x_ba.append(self.gen_a.decode(h_b))
			x_ab.append(self.gen_b.decode(h_a))
		x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
		x_ba = torch.cat(x_ba)
		x_ab = torch.cat(x_ab)
		self.train()
		return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

	def dis_update(self, x_a, x_b, hyperparameters):
		self.dis_opt.zero_grad()
		# encode
		h_a, n_a = self.gen_a.encode(x_a)
		h_b, n_b = self.gen_b.encode(x_b)
		# decode (cross domain)
		x_ba = self.gen_a.decode(h_b + n_b)
		x_ab = self.gen_b.decode(h_a + n_a)
		# D loss
		self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
		self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
		self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
		self.loss_dis_total.backward()
		self.dis_opt.step()

	def update_learning_rate(self):
		if self.dis_scheduler is not None:
			self.dis_scheduler.step()
		if self.gen_scheduler is not None:
			self.gen_scheduler.step()

	def resume(self, checkpoint_dir, hyperparameters):
		# Load generators
		last_model_name = get_model_list(checkpoint_dir, "gen")
		state_dict = torch.load(last_model_name)
		self.gen_a.load_state_dict(state_dict['a'])
		self.gen_b.load_state_dict(state_dict['b'])
		iterations = int(last_model_name[-11:-3])
		# Load discriminators
		last_model_name = get_model_list(checkpoint_dir, "dis")
		state_dict = torch.load(last_model_name)
		self.dis_a.load_state_dict(state_dict['a'])
		self.dis_b.load_state_dict(state_dict['b'])
		# Load optimizers
		state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
		self.dis_opt.load_state_dict(state_dict['dis'])
		self.gen_opt.load_state_dict(state_dict['gen'])
		# Reinitilize schedulers
		self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
		self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
		print('Resume from iteration %d' % iterations)
		return iterations

	def save(self, snapshot_dir, iterations):
		# Save generators, discriminators, and optimizers
		gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
		dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
		opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
		torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict(), 'task': self.netF.state_dict()}, gen_name)
		torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
		torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

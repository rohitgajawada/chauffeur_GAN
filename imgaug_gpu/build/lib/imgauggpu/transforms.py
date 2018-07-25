import time

import numpy as np

import imgaug as ia
import torch


from . import functional as F



def do_assert(condition, message="Assertion failed."):
	"""
	Function that behaves equally to an `assert` statement, but raises an
	Exception.

	This is added because `assert` statements are removed in optimized code.
	It replaces `assert` statements throughout the library that should be
	kept even in optimized code.

	Parameters
	----------
	condition : bool
		If False, an exception is raised.

	message : string, optional(default="Assertion failed.")
		Error message.

	"""
	if not condition:
		raise AssertionError(str(message))


class Sometimes(object):

	def __init__(self, transform, p ):

		self.transform = transform

		self.p = ia.parameters.Binomial(p)



	def __call__(self, images):

		nb_images = len(images) # 120 x 3 x H x W
		seeds = ia.current_random_state().randint(0, 10**6, (nb_images,))
		rs_image = ia.new_random_state(seeds[0])

		samples= self.p.draw_samples((nb_images,), random_state=rs_image)
		samples_true_index = np.where(samples == 1)[0]
		samples_false_index = np.where(samples == 0)[0]


		# Check for the zero case , related to pytorch isuee of not accepting zero size array

		if samples_true_index.size > 0:
			samples_true_index = torch.cuda.LongTensor(samples_true_index)
			images_to_transform = torch.index_select(images, 0, samples_true_index)
			if samples_false_index.size > 0:

				# images_not_to_transform = torch.index_select(images, 0, torch.cuda.LongTensor(
				#     samples_false_index))

				transformed_imgs = self.transform(images_to_transform)
				images = images.index_copy_(0, samples_true_index, transformed_imgs)
				return images
				# return torch.cat((self.transform(images_to_transform), images_not_to_transform), 0)

			else:
				return self.transform(images)

		else:
			return images





class Add(ia.augmenters.Add):

	"""
		Wrapper for adding class from imgaug

	"""
	def __call__(self, images):
		# input_dtypes = copy_dtypes_for_restore(images, force_list=True)
		result = images
		nb_images = len(images)
		# Generate new seeds con
		seeds = ia.current_random_state().randint(0, 10**6, (nb_images,))
		rs_image = ia.new_random_state(seeds[0])
		per_channel = self.per_channel.draw_sample(random_state=rs_image)

		if per_channel == 1:
			nb_channels = images.shape[1]

			for c in range(nb_channels):
				samples = self.value.draw_samples((nb_images, 1, 1, 1), random_state=rs_image).astype(
					np.float32)
				do_assert(samples.all() >= 0)

				result[:, c:(c+1), ...] = F.add(images[:, c:(c+1), ...], torch.cuda.FloatTensor(samples))

		else:
			samples = self.value.draw_samples((nb_images, 1, 1, 1), random_state=rs_image).astype(np.float32)
			do_assert(samples.all() >= 0)

			result = F.add(images, torch.cuda.FloatTensor(samples))

		# image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
		# image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

		return result




class Multiply(ia.augmenters.Multiply):

	"""
		Wrapper for multiply class from imgaug

	"""
	def __call__(self, images):

		result = images
		nb_images = len(images)
		# Generate new seeds con
		seeds = ia.current_random_state().randint(0, 10**6, (nb_images,))
		rs_image = ia.new_random_state(seeds[0])
		per_channel = self.per_channel.draw_sample(random_state=rs_image)

		if per_channel == 1:
			nb_channels = images.shape[1] #Considering (N C H W)

			for c in range(nb_channels):
				samples = self.mul.draw_samples((nb_images, 1, 1, 1), random_state=rs_image).astype(
					np.float32)
				do_assert(samples.all() >= 0)

				result[:, c:(c+1), ...] = F.multiply(images[:, c:(c+1), ...], torch.cuda.FloatTensor(samples))

		else:
			samples = self.mul.draw_samples((nb_images, 1, 1, 1), random_state=rs_image).astype(np.float32)
			do_assert(samples.all() >= 0)
			result = F.multiply(images, torch.cuda.FloatTensor(samples))

		return result






class MultiplyElementwise(ia.augmenters.MultiplyElementwise):
	"""
	Wrapper
	"""

	def __init__(self, mul=1.0, per_channel=False, name=None, deterministic=False, random_state=None):
		super(MultiplyElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
		self.prob = mul

		"""
		if ia.is_single_number(mul):
			assert mul >= 0.0, "Expected multiplier to have range [0, inf), got value %.4f." % (mul,)
			self.mul = ia.parameters.Deterministic(mul)
		elif ia.is_iterable(mul):
			assert len(mul) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(mul),)
			self.mul = ia.parameters.Uniform(mul[0], mul[1])
		elif isinstance(mul, ia.parameters.StochasticParameter):
			self.mul = mul
		else:
			raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(mul),))

		if per_channel in [True, False, 0, 1, 0.0, 1.0]:
			self.per_channel = ia.parameters.Deterministic(int(per_channel))
		elif ia.is_single_number(per_channel):
			assert 0 <= per_channel <= 1.0
			self.per_channel = ia.parameters.Binomial(per_channel)
		else:
			raise Exception("Expected per_channel to be boolean or number or StochasticParameter")
		"""


	def __call__(self, images):

		"""
		nb_images = len(images)


		nb_channels, height, width = images[0].shape

		seeds = ia.current_random_state().randint(0, 10**6, (nb_images,))
		rs_image = ia.new_random_state(seeds[0])


		per_channel = self.per_channel.draw_sample(random_state=rs_image)
		if per_channel == 1:

			samples = self.mul.draw_samples((nb_images, height, width, nb_channels), random_state=rs_image)


		else:
			capture_time = time.time()
			samples = self.mul.draw_samples((nb_images,  height, width, 1), random_state=rs_image)
			print ("Sampling time ", time.time() - capture_time)
			print(samples.shape)

			capture_time = time.time()
			samples = np.tile(samples, (1, 1, 1, nb_channels))
			print ("Tiling time ", time.time() - capture_time)
			print (samples.shape)

		samples = np.swapaxes(samples, 1, 3)
		samples = np.swapaxes(samples, 2, 3)
		"""


		return F.multiply_element_wise(images, self.prob)



def Dropout(p=0, per_channel=False, name=None, deterministic=False,
			random_state=None):
	"""
		Copy of IMGAUG dropout,
		TODO: how to I basically make imgaug dropout to call my multiply elemenwise ?
	"""
	if ia.is_single_number(p):
		p2 = ia.parameters.Binomial(1 - p)
	elif isinstance(p, (tuple, list)):
		do_assert(len(p) == 2)
		do_assert(p[0] < p[1])
		do_assert(0 <= p[0] <= 1.0)
		do_assert(0 <= p[1] <= 1.0)
		p2 = ia.parameters.Binomial(ia.parameters.Uniform(1 - p[1], 1 - p[0]))
	else:
		raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))
	return MultiplyElementwise(p2, per_channel=per_channel, name=name, deterministic=deterministic, random_state=random_state)




def CoarseDropout(p=0, size_px=None, size_percent=None,
				  per_channel=False, min_size=4, name=None, deterministic=False,
				  random_state=None):
	"""
	The version exactly the same, it would be nice to just route the calling from IMGaug
	"""
	if ia.is_single_number(p):
		p2 = ia.parameters.Binomial(1 - p)
	elif ia.is_iterable(p):
		do_assert(len(p) == 2)
		do_assert(p[0] < p[1])
		do_assert(0 <= p[0] <= 1.0)
		do_assert(0 <= p[1] <= 1.0)
		p2 = ia.parameters.Binomial(ia.parameters.Uniform(1 - p[1], 1 - p[0]))
	elif isinstance(p, ia.parameters.StochasticParameter):
		p2 = p
	else:
		raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))

	if size_px is not None:
		p3 = ia.parameters.FromLowerResolution(other_param=p2, size_px=size_px, min_size=min_size)
	elif size_percent is not None:
		p3 = ia.parameters.FromLowerResolution(other_param=p2, size_percent=size_percent, min_size=min_size)
	else:
		raise Exception("Either size_px or size_percent must be set.")


	return MultiplyElementwise(p3, per_channel=per_channel, name=name, deterministic=deterministic, random_state=random_state)




class ContrastNormalization(ia.augmenters.ContrastNormalization):



	def __call__(self, images):

		result = images
		nb_images = len(images)
		# Generate new seeds con
		seeds = ia.current_random_state().randint(0, 10**6, (nb_images,))
		rs_image = ia.new_random_state(seeds[0])
		per_channel = self.per_channel.draw_sample(random_state=rs_image)

		if per_channel == 1:
			nb_channels = images.shape[1] #Considering (N C H W)

			for c in range(nb_channels):
				alphas = self.alpha.draw_samples((nb_images, 1, 1, 1), random_state=rs_image).astype(
					np.float32)
				do_assert(alphas.all() >= 0)

				result[:, c:(c+1), ...] = F.contrast(images[:, c:(c+1), ...], torch.cuda.FloatTensor(alphas))

		else:
			alphas = self.alpha.draw_samples((nb_images, 1, 1, 1), random_state=rs_image).astype(np.float32)
			do_assert(alphas.all() >= 0)

			#print(alphas)

			result = F.contrast(images, torch.cuda.FloatTensor(alphas))
		return result






class GaussianBlur(ia.augmenters.GaussianBlur): # pylint: disable=locally-disabled, unused-variable, line-too-long



	def __call__ (self, images):

		result = images
		nb_images = len(images)

		nb_channels, height, width = images[0].shape

		seeds = ia.current_random_state().randint(0, 10**6, (nb_images,))
		rs_image = ia.new_random_state(seeds[0])

		samples = self.sigma.draw_samples((nb_images,), random_state=rs_image)



		# note that while gaussian_filter can be applied to all channels
		# at the same time, that should not be done here, because then
		# the blurring would also happen across channels (e.g. red
		# values might be mixed with blue values in RGB)
		for c in range(nb_channels):

			result[:, c:(c+1), ...] = F.blur(result[:, c:(c+1), ...], samples)

		return result




class AddElementwise(ia.augmenters.AddElementwise):



	def __init__(self, value=0, per_channel=False, name=None, deterministic=False, random_state=None):
		"""Create a new AddElementwise instance.


		"""
		self.value = value

	def __call__(self, images):

		"""
		nb_images = len(images)


		nb_channels, height, width = images[0].shape

		seeds = ia.current_random_state().randint(0, 10**6, (nb_images,))
		rs_image = ia.new_random_state(seeds[0])

		per_channel = self.per_channel.draw_sample(random_state=rs_image)
		if per_channel == 1:
			samples = self.value.draw_samples((nb_images, height, width, nb_channels), random_state=rs_image)


		else:
			samples = self.value.draw_samples((nb_images,  height, width, 1), random_state=rs_image)

			samples = np.tile(samples, (1, 1, 1, nb_channels))


		samples = np.swapaxes(samples, 1, 3)
		samples = np.swapaxes(samples, 2, 3)
		"""

		return F.add_element_wise(images, self.value)





def AdditiveGaussianNoise(loc=0, scale=0, per_channel=False, name='None', deterministic=False, random_state=None):

	if ia.is_single_number(loc):
		loc2 = ia.parameters.Deterministic(loc)
	elif ia.is_iterable(loc):
		assert len(loc) == 2, "Expected tuple/list with 2 entries for argument 'loc', got %d entries." % (len(scale),)
		loc2 = ia.parameters.Uniform(loc[0], loc[1])
	elif isinstance(loc, ia.parameters.StochasticParameter):
		loc2 = loc
	else:
		raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter for argument 'loc'. Got %s." % (type(loc),))

	if ia.is_single_number(scale):
		scale2 = ia.parameters.Deterministic(scale)
	elif ia.is_iterable(scale):
		assert len(scale) == 2, "Expected tuple/list with 2 entries for argument 'scale', got %d entries." % (len(scale),)
		scale2 = ia.parameters.Uniform(scale[0], scale[1])
	elif isinstance(scale, ia.parameters.StochasticParameter):
		scale2 = scale
	else:
		raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter for argument 'scale'. Got %s." % (type(scale),))

	return AddElementwise(scale,
						  per_channel=per_channel, name=name,
						  deterministic=deterministic, random_state=random_state)




class ChangeColorspace(ia.augmenters.ChangeColorspace):
	"""
	Augmenter to change the colorspace of images.

	"""


	def __call__(self, images):



		# Convert to Grayscale direction

		nb_images = len(images)
		# Generate new seeds con
		seeds = ia.current_random_state().randint(0, 10**6, (nb_images,))
		rs_image = ia.new_random_state(seeds[0])

		samples = self.alpha.draw_samples((nb_images, 1, 1, 1), random_state=rs_image).astype(np.float32)
		do_assert(samples.all() >= 0)
		result = F.grayscale(images, torch.cuda.FloatTensor(samples))

		return result




# TODO tests
# TODO rename to Grayscale3D and add Grayscale that keeps the image at 1D
def Grayscale(alpha=0, from_colorspace="RGB", name=None,
			  deterministic=False, random_state=None):


	return ChangeColorspace(to_colorspace=ChangeColorspace.GRAY, alpha=alpha,
							from_colorspace=from_colorspace, name=name,
							deterministic=deterministic, random_state=random_state)

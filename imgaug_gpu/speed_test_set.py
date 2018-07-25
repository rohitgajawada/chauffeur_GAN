from imgauggpu import Sometimes

def simple_operations(using_gpu):
	"""
	Returns a vector of callable classes. YOu can after instance an augmenter class with
	this vector as a parameter, and you will have a callable augmenter !
	Returns
		A vector of callable augmeters
	"""

	if using_gpu:
		import imgauggpu as iag
	else:
		import imgaug.augmenters as iag

	st = lambda aug: iag.Sometimes(aug, 0.4)
	oc = lambda aug: iag.Sometimes(aug, 0.3)
	rl = lambda aug: iag.Sometimes(aug, 0.09)
	return [rl(iag.ContrastNormalization((0.5, 1.5), per_channel=0.5)), rl(iag.GaussianBlur((0, 1.5))),rl(iag.Grayscale((0.0, 1))),
			oc(iag.Add((-20, 20))), oc(iag.Multiply((0.10, 1.5), per_channel=0.2)), # Arithmetic
			oc(iag.Dropout((0,0.1), per_channel=0.5)), oc(iag.CoarseDropout((0, 0.1), size_percent=(0.08, 0.2), per_channel=0.5, size_px=(16, 16)))]  # Dropout
		   # iag.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.10*255), name='Additive')]  # Gaussian noise

def complex_operations(using_gpu):
	"""
	Same as simple operations but also adding gaussian blur and contrast normalization
	Returns
		 A vector of callable augmeters
	"""
	if using_gpu:
		import imgauggpu as iag
	else:
		import imgaug.augmenters as iag


	return simple_operations() + \
		   [iag.GaussianBlur(sigma=(0.0, 3.0)),
			iag.ContrastNormalization((0.5, 1.5))]

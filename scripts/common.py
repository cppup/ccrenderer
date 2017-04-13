from __future__ import division
import os
from PIL import Image, ImageDraw
import numpy as np

import logging

__all__ = ['log', 'Vec2i', 'Vec2f', 'Vec3i', 'Vec3f', 'get_color', 'MyImage']


def init_logger(name):
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	if not len(logger.handlers):
		stream = logging.StreamHandler()
		stream.setFormatter(logging.Formatter(
			fmt='[%(levelname)s] %(asctime)s %(message)s',
			datefmt='%H:%M:%S')
		)
		logger.addHandler(stream)
	return logger


os.chdir(os.path.dirname(os.path.abspath(__file__)))  # change path/to/script/base/dir
log = init_logger(__name__)
log.info('pwd: {}'.format(os.getcwd()))


# ===============================================
#  types
# -----------------------------------------------


class MyImage(object):

	def __init__(self, size, mode, color=0):
		if isinstance(size, tuple):
			if mode == 'RGBA':
				color = (0, 0, 0, 255)
			self.image = Image.new(mode, size, color)
		else:
			fp = size
			self.image = Image.open(fp, mode='r')
		self._init_attr()

	def _init_attr(self):
		self.draw = ImageDraw.Draw(self.image)
		self.width = self.image.width
		self.height = self.image.height
	
	def set(self, pt, color):
		self.draw.point(pt, color)
	
	def flip_vertically(self):
		self.image = self.image.transpose(Image.FLIP_TOP_BOTTOM)
	
	def save(self, filename):
		self.image.save(filename)
		
	def __getattr__(self, name):
		if not hasattr(self, name):
			return getattr(self.image, name)
		return super(MyImage, self).__getattribute__(self, name)


class _Vec(object):

	_fields = []

	def __init__(self, *arg):
		# self.x = None
		# self.y = None
		# self.z = None
		self.x = self.y = self.z = 0
		for k, v in zip(self._fields, arg[0]):
			setattr(self, k, v)

	def __eq__(self, other):
		# return tuple(getattr(self, k) for k in self._fields) == other
		return tuple(self) == tuple(other)
	
	def __lt__(self, other):
		return tuple(self) < tuple(other)

	def __getitem__(self, index):
		return getattr(self, self._fields[index])

	def __repr__(self):
		return '{}{}'.format(self.__class__.__name__, tuple(getattr(self, k) for k in self._fields))

	def __len__(self):
		return len(self._fields)

	def __div__(self, v):
		return self.__mul__(1. / v)
	
	def __truediv__(self, v):
		return self.__mul__(1. / v)

	def normalize(self):
		norm = abs(self)
		return self.__class__(*[(x / norm) for x in self])


class _Vec2(_Vec):

	_fields = ['x', 'y']

	def __add__(self, v):
		return self.__class__(self.x + v.x, self.y + v.y)

	def __sub__(self, v):
		return self.__class__(self.x - v.x, self.y - v.y)

	def __xor__(self, v):
		return Vec3i(0, 0, self.x * v.y - self.y * v.x)

	def __mul__(self, v):
		if isinstance(v, (int, float)):
			return self.__class__(self.x * v, self.y * v)
		return self.x * v.x - self.y * v.y

	def __abs__(self):
		return (self.x ** 2 + self.y ** 2) ** 0.5


class _Vec3(_Vec):

	_fields = ['x', 'y', 'z']

	def __add__(self, v):
		return self.__class__(self.x + v.x, self.y + v.y, self.z + v.z)

	def __sub__(self, v):
		return self.__class__(self.x - v.x, self.y - v.y, self.z - v.z)

	def __xor__(self, v):
		u1, u2, u3 = self.x, self.y, self.z
		v1, v2, v3 = v.x, v.y, v.z
		return self.__class__(u2 * v3 - u3 * v2, u3 * v1 - u1 * v3, u1 * v2 - u2 * v1)

	def __mul__(self, v):
		if isinstance(v, (int, float)):
			return self.__class__(self.x * v, self.y * v, self.z * v)
		u1, u2, u3 = self.x, self.y, self.z
		v1, v2, v3 = v.x, v.y, v.z
		return u1 * v1 + u2 * v2 + u3 * v3

	def __abs__(self):
		return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5


class Vec2i(_Vec2):

	def __init__(self, *arg):
		if len(arg) == 1:
			arg = arg[0]
		arg = map(int, arg)
		super(Vec2i, self).__init__(arg)


class Vec3i(_Vec3):

	def __init__(self, *arg):
		if len(arg) == 1:
			arg = arg[0]
		arg = map(int, arg)
		super(Vec3i, self).__init__(arg)


class Vec2f(_Vec2):

	def __init__(self, *arg):
		if len(arg) == 1:
			arg = arg[0]
		arg = map(float, arg)
		super(Vec2f, self).__init__(arg)


class Vec3f(_Vec3):

	def __init__(self, *arg):
		if len(arg) == 1:
			arg = arg[0]
		arg = map(float, arg)
		super(Vec3f, self).__init__(arg)


# ===============================================
#  methods
# -----------------------------------------------

def get_color(intensity, color=None):
	if color:
		r, g, b = color[:3]
	else:
		r = g = b = 255
	return tuple(int(round(x))
				 for x in
				 (intensity * r, intensity * g, intensity * b, 255))


def test():
	# test init
	Vec2i() == (0, 0)
	Vec3i() == (0, 0, 0)

	# test type cast	
	assert Vec2i((1.1, 2.6)) == (1, 2) == Vec2i((1.6, 2.1))

	# test add & subtract
	a = Vec2i(1.2, 2)
	b = Vec2i(3, 4)
	c = Vec2i(7, 10)
	assert (a + b) == (4, 6) != (1, 1), a + b
	assert (c - b) == (4, 6) != (1, 1), c - b

	# test normalize & abs
	a = Vec3f(3, 4, 5)
	n = a.normalize()
	assert n == Vec3f(0.4242640687119285, 0.565685424949238,
					  0.7071067811865475), 'normalize failed'
	assert abs(n) == 1.0, 'vector module failed'

	# test mul & div
	a *= .5
	assert a == Vec3f(1.5, 2.0, 2.5)

	a /= 2.
	assert a == Vec3f(0.75, 1.0, 1.25)

	# test unpack
	x, y, z = a
	assert x == 0.75 and y == 1.0 and z == 1.25

	# test min & max
	a = Vec3f(1, 1, 2)
	b = Vec3f(1, 2, 1)
	assert a < b, '{}'.format(a - b)
	assert min(a, b) == a, '{}'.format(a - b)
	assert max(a, b) > a, '{}'.format(a - b)


if __name__ == '__main__':
	test()

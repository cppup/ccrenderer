import sys
from common import *
from line import *


def wire_triangle(t0, t1, t2, image, color):
	"""not filled"""
	# sort vertices, t0, t1, t2 lower to upper (using bubble sort)
	if t0.y > t1.y:
		t0, t1 = t1, t0
	if t1.y > t2.y:
		t1, t2 = t2, t1
	if t0.y > t1.y:
		t0, t1 = t1, t0
	line(t0, t1, image, green)
	line(t1, t2, image, green)
	line(t2, t0, image, red)

# -----------------------------------------------------


def _fill_half(t0, t1, t2, image, color):
	"""
	# XXX: border check:
		1) t1.y == t0.y, that is, horizontal line
		2) t2.y == t0.y, that is, three points are in one line, then there is no triangle
	"""
	alpha = float(t2.x - t0.x) / (t2.y - t0.y)
	# be careful with divisions by zero
	beta = float(t1.x - t0.x) / (t1.y - t0.y)

	if t0.y > t1.y:
		a, b = t1.y, t0.y
	else:
		a, b = t0.y, t1.y

	for y in range(a, b + 1):
		Ax = int((y - t0.y) * alpha + t0.x)
		Bx = int((y - t0.y) * beta + t0.x)
		if Ax > Bx:
			Ax, Bx = Bx, Ax
		for x in range(Ax, Bx + 1):
			image.set((x, y), color)


def old_triangle(t0, t1, t2, image, color):
	"""implemented by sweeping line (y-scan line)"""

	# sort vertices, t0, t1, t2 lower to upper (using bubble sort)
	if t0.y > t1.y:
		t0, t1 = t1, t0
	if t1.y > t2.y:
		t1, t2 = t2, t1
	if t0.y > t1.y:
		t0, t1 = t1, t0

	# draw first (bottom) half of the triangle: calculate x of both left side
	# and right side
	_fill_half(t0, t1, t2, image, color)

	# draw second (upper) half of the triangle: calculate x of both left side
	# and right side
	_fill_half(t2, t1, t0, image, color)


# -----------------------------------------------------

def find_bounding_box(points):
	xmax = xmin = points[0].x
	ymax = ymin = points[0].y
	for p in points:
		x, y = p.x, p.y
		if xmax < x:
			xmax = x
		if xmin > x:
			xmin = x
		if ymax < y:
			ymax = y
		if ymin > y:
			ymin = y

	# XXX: FIX ME. float err cause border error
	# assert xmax >= 800 or ymax >= 800, 'xmax={}, ymax={}'.format(xmax, ymax)
	# xmin = max(0, xmin)
	# ymin = max(0, ymin)
	# xmax = min(width - 1, xmax)
	# ymax = min(height - 1, ymax)
	return xmin, ymin, xmax, ymax


def iter_pixel_of_box(bbox):
	x0, y0, x1, y1 = bbox
	while y0 <= y1:
		x = x0
		while x <= x1:
			yield Vec2i(x, y0)
			x += 1
		y0 += 1


def barycentric(points, pt):
	P = pt
	A, B, C = points
	AB = B - A
	AC = C - A
	PA = A - P

	xs = [float(x.x) for x in (AB, AC, PA)]
	ys = [float(x.y) for x in (AB, AC, PA)]

	a = Vec3f(*xs)
	b = Vec3f(*ys)
	u, v, p = a ^ b

	# from numpy import array, cross
	# a = array(xs)
	# b = array(ys)
	# u, v, p = cross(a, b)

	# log.debug('{} {} {}'.format(u, v, p))

	if p == 0:
		return Vec3f(-1, 1, 1)  # triangle is degenerated
	return Vec3f(1. - (u + v) / p, u / p, v / p)


def inside(points, pt):
	# P = complex(*pt)
	# PA, PB, PC = [complex(*x) - P for x in points]
	# t1 = (PA * PB).imag
	# t2 = (PB * PC).imag
	# t3 = (PC * PA).imag
	# if (t1 > 0 and t2 > 0 and t3 > 0) or (t1 < 0 and t2 < 0 and t3 < 0):
	# 	return True
	# return False

	bc = barycentric(points, pt)
	if (bc.x < 0 or bc.y < 0 or bc.z < 0):
		return False
	return True


def triangle(*arg):
	# parse arg
	if len(arg) == 3:
		(t0, t1, t2), image, color = arg
	elif len(arg) == 5 and isinstance(arg[4], float):
		return texture_triangle(*arg)
	else:
		t0, t1, t2, image, color = arg

	# use z buffer if vec3
	if isinstance(t0, (Vec3i, Vec3f)):
		return zbuffer_triangle(t0, t1, t2, image, color)

	points = t0, t1, t2
	bbox = find_bounding_box(points)
	for pt in iter_pixel_of_box(bbox):
		if inside(points, pt):
			image.set(pt, color)


class _ZBuffer(list):

	def __init__(self, width=0, height=0):
		self.width = width
		self.height = height
		super(_ZBuffer, self).__init__([-sys.maxint] * width * height)

	def __getitem__(self, key):
		if isinstance(key, tuple):
			key = int(key[0] + key[1] * self.width)
		return super(_ZBuffer, self).__getitem__(key)

	def __setitem__(self, key, value):
		if isinstance(key, tuple):
			key = int(key[0] + key[1] * self.width)
		return super(_ZBuffer, self).__setitem__(key, value)

	def __repr__(self):
		return '<_ZBuffer: ({}x{})>'.format(self.width, self.height)

	def set(self, width, height):
		self.__init__(width, height)


g_zbuffer = _ZBuffer()


def zbuffer_triangle(t0, t1, t2, image, color):
	global g_zbuffer

	points = t0, t1, t2
	bbox = find_bounding_box(points)
	# log.debug('bbox: {}'.format(bbox))

	for pt in iter_pixel_of_box(bbox):
		bc = barycentric(points, Vec3f(pt.x, pt.y, 0))
		if (bc.x < 0 or bc.y < 0 or bc.z < 0):
			continue
		z = sum(points[i].z * bc[i] for i in range(3))

		if g_zbuffer[pt.x, pt.y] < z:
			g_zbuffer[pt.x, pt.y] = z

			if len(color) == 3:  # three vertex color
				c1, c2, c3 = color
				color = tuple([int(round(
					c1[i] * bc[0] + c2[i] * bc[1] + c3[i] * bc[2])) for i in range(3)] + [255])
				# color = tuple(map(int, map(round, color[0])))
			image.set(pt, color)

import numpy as np
g_camera = Vec3f(0, 0, 1)
g_mvp = None

def get_mvp():
	global g_mvp
	if g_mvp is not None:
		return g_mvp
	c = np.array(g_camera, 'f')
	view = c!=0
	c[view] = -1 / c[view]
	g_mvp = np.array([
		[1,0,0,0],
		[0,1,0,0],
		[0,0,1,0],
		list(c) + [1]
	])
	return g_mvp

def perspective(v):
	v = [v.x, v.z, v.z, 1]
	mvp = get_mvp()
	retro = sum(mvp[3] * v)
	v2 = mvp.dot(v) / retro
	return Vec3i(v2[:3])


def texture_triangle(points, image, uv_coords, model, intensity):
	global g_zbuffer

	bbox = find_bounding_box(points)
	cnt = len(points)
	for pt in iter_pixel_of_box(bbox):
		bc = barycentric(points, pt)
		
		if (bc.x < 0 or bc.y < 0 or bc.z < 0):
			continue
		
		z = sum(points[i].z * bc[i] for i in range(cnt))

		if bc.x != 0 and bc.y != 0 and bc.z != 0:
			continue

		image.set(pt, green)

		if g_zbuffer[pt.x, pt.y] < z:
			g_zbuffer[pt.x, pt.y] = z
			
			# uv = sum((uv_coords[i] * bc[i] for i in range(cnt)), Vec2f(0, 0))
			# # assert uv[0] <= 1 and uv[1] <=1, 'uv={}, bc={}'.format(uv, bc)

			# color = model.diffuse(uv)
			# color = get_color(intensity, color)
			# image.set(pt, color)


def main():
	image = MyImage((200, 200), 'RGBA')

	t0 = [Vec2i(10, 70),   Vec2i(50, 160),  Vec2i(70, 80)]
	t1 = [Vec2i(180, 50),  Vec2i(150, 1),   Vec2i(70, 180)]
	t2 = [Vec2i(180, 150), Vec2i(120, 160), Vec2i(130, 180)]
	triangle(t0[0], t0[1], t0[2], image, red)
	triangle(t1[0], t1[1], t1[2], image, white)
	triangle(t2[0], t2[1], t2[2], image, green)

	image.flip_vertically()
	image.save('./output/triangle.png')


if __name__ == '__main__':
	main()

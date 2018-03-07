from __future__ import division
from common import *
import numpy as np


_Viewport = None
_Projection = None
_ModelView = None


class Matrix:

	@classmethod
	def identity(cls, N=4):
		return np.eye(N)

	@classmethod
	def embed(cls, v, N=4):
		v2 = np.zeros(N)
		v2[:len(v)] = v
		v2[-1] = 1
		return v2.reshape(N, 1)


class Shader(object):

	def __init__(self, model, light_dir):
		self.model = model
		self.light_dir = light_dir
		self._Trasform = _Viewport.dot(_Projection.dot(_ModelView))

	def vertex(self, iface, ivert):
		raise NotImplementedError

	def fragment(self, bc, color):
		raise NotImplementedError


class GouraudShader(Shader):

	def __init__(self, model, light_dir):
		super(GouraudShader, self).__init__(model, light_dir)
		# written by vertex shader, read by fragment shader
		self.varying_intensity = Vec3f()

	def vertex(self, iface, ivert):
		self.varying_intensity[ivert] = max(0., self.model.normal(iface, ivert) * self.light_dir)
		gl_vertex = Matrix.embed(self.model.vert(iface, ivert))  # Homogeneous coordinates
		v = np.dot(self._Trasform, gl_vertex) # dot(4x4 Matrix, 4x1 Matrix) = 4x1 Matrix
		v = (v / v[3])
		return Vec3i(v.flatten())

	def fragment(self, bc, color):
		intensity = self.varying_intensity * bc
		color = tuple([int(intensity * 255)] * 3)
		return False, color


class PhongShader(Shader):

	def __init__(self, model, light_dir):
		super(PhongShader, self).__init__(model, light_dir)
		# written by vertex shader, read by fragment shader
		self.varying_intensity = Vec3f()
		self.varying_uv = [None] * 3
	
	def vertex(self, iface, ivert):
		self.varying_uv[ivert] = self.model.uv(iface, ivert)
		self.varying_intensity[ivert] = max(0., self.model.normal(iface, ivert) * self.light_dir)
		gl_vertex = Matrix.embed(self.model.vert(iface, ivert))  # Homogeneous coordinates
		v = np.dot(self._Trasform, gl_vertex) # dot(4x4 Matrix, 4x1 Matrix) = 4x1 Matrix
		v = (v / v[3])
		return Vec3i(v.flatten())
	
	def fragment(self, bc, color):
		intensity = self.varying_intensity * bc
		uv = Vec2i()
		for i, (u, v) in enumerate(self.varying_uv):
			uv.x += u * bc[i]
			uv.y += v * bc[i]
		# color = self.model.diffuse(uv) * intensity
		color = get_color(intensity, self.model.diffuse(uv))
		return False, color


def viewport(x, y, w, h):
	global _Viewport
	v = Matrix.identity()
	v[:3] = [
			[w * .5, 0, 0, x + w * .5],
			[0, h * .5, 0, y + h * .5],
			[0, 0, 255 * .5, 255 * .5]
	]
	_Viewport = v
	return v


def projection(coeff):
	global _Projection
	p = Matrix.identity()
	p[3][2] = coeff
	_Projection = p
	return p


def lookat(eye, center, up):
	global _ModelView
	z = (eye - center).normalize()
	x = (up ^ z).normalize()
	y = (z ^ x).normalize()
	_ModelView = Matrix.identity()
	for i in range(3):
		_ModelView[0][i] = x[i]
		_ModelView[1][i] = y[i]
		_ModelView[2][i] = z[i]
		_ModelView[i][3] = -center[i]
	return _ModelView


def barycentric(points, pt):
	A, B, C = points
	AB = B - A
	AC = C - A
	PA = A - pt

	# a = Vec3f(AB.x, AC.x, PA.x)
	# b = Vec3f(AB.y, AC.y, PA.y)
	# u, v, p = a ^ b

	# if p == 0:  # triangle is degenerated
	# 	return Vec3f(-1, 1, 1)
	# return Vec3f(1. - (u + v) / p, u / p, v / p)

	a = np.array((AB.x, AC.x, PA.x))
	b = np.array((AB.y, AC.y, PA.y))
	c = np.cross(a, b)

	if c[2] != 0:
		u, v, _ = c / c[2]
		return Vec3f(1. - u - v, u, v)
	return Vec3f(-1, 1, 1)


def _find_bounding_box(points):
	xs = [p.x for p in points]
	ys = [p.y for p in points]
	xmin, ymin = min(xs), min(ys)
	xmax, ymax = max(xs), max(ys)
	# log.debug('bbox: {}'.format((xmin, ymin, xmax, ymax)))
	return xmin, ymin, xmax, ymax


def _iter_pixel_of_bbox(x0, y0, x1, y1):
	pt = Vec3i()
	pt.y = y0
	while pt.y <= y1:
		pt.x = x0
		while pt.x <= x1:
			yield pt
			pt.x += 1
		pt.y += 1


def triangle(points, shader, image, zbuffer):

	bbox = _find_bounding_box(points)
	for P in _iter_pixel_of_bbox(*bbox):
		bc = barycentric(points, P)

		if (bc.x < 0 or bc.y < 0 or bc.z < 0):  # out of triangle
			continue

		# eps = 1e-2
		# if (abs(bc.x) <= eps or abs(bc.y) <= eps or abs(bc.z) <= eps):
		# 	image.set((P.x, P.y), green)

		P.z = int(sum(points[i].z * bc[i] for i in range(3)) + .5)

		xy = (P.x, P.y)
		if zbuffer.get(xy) > P.z:
			continue

		color = None
		discard, color = shader.fragment(bc, color)

		if not discard:
			zbuffer.set(xy, P.z)
			image.set(xy, color)


from line import *
def wire_triangle(points, shader, image, zbuffer):
	"""not filled"""
	t0, t1, t2 = points
	# sort vertices, t0, t1, t2 lower to upper (using bubble sort)
	if t0.y > t1.y:
		t0, t1 = t1, t0
	if t1.y > t2.y:
		t1, t2 = t2, t1
	if t0.y > t1.y:
		t0, t1 = t1, t0
	line(t0, t1, image, green)
	line(t1, t2, image, green)
	line(t2, t0, image, green)


def main():
	from model import Model

	model = Model('../obj/african_head.obj')

	light_dir = Vec3f(1, 1, 1)
	eye = Vec3f(1, 1, 3)
	center = Vec3f(0, 0, 0)
	up = Vec3f(0, 1, 0)

	# light_dir = Vec3f(0, 0, -1)
	# eye = Vec3f(0, 0, 3)
	# center = Vec3f(0, 0, 0)
	# up = Vec3f(0, 1, 0)

	width, height = 800, 800
	image = MyImage((width, height), 'RGBA')
	zbuffer = MyImage((width, height), 'L')

	log.debug('image: {}'.format(image))

	lookat(eye, center, up)
	viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4)
	projection(-1. / abs(eye - center))
	light_dir = light_dir.normalize()

	# log.debug('Transform:\nModelView=\n{}\nProjection=\n{}\nViewport=\n{}'.format(_ModelView, _Projection, _Viewport))
	# log.debug('all in one=\n{:}'.format(_Viewport.dot(_Projection.dot(_ModelView))))

	# shader = GouraudShader(model, light_dir)
	shader = PhongShader(model, light_dir)
	for i in range(model.nfaces()):
		screen_coords = [shader.vertex(i, j) for j in range(3)]
		# log.debug('screen_coords: {}'.format(screen_coords))
		triangle(screen_coords, shader, image, zbuffer)

	image.flip_vertically()
	zbuffer.flip_vertically()
	image.save('./output/mygl.png')
	zbuffer.save('./output/zbuffer.png')


if __name__ == '__main__':
	import time
	now = time.time()
	main()
	delta = time.time() - now
	log.debug('it take {:.3f} seconds to run.'.format(delta))

from common import *
from line import *
import numpy as np


def viewport(x, y, w, h):
	# w -= 1
	# h -= 1
	s = np.array([
		[.5, 0, 0],
		[0, .5, 0],
		[0, 0, 1]
	])
	m = np.array([
		[w, 0, x + w * .5],
		[0, h, y + h * .5],
		[0, 0, 1]
	])
	return m.dot(s)


def dot(a, b):
	c = np.dot(a, b)
	log.debug('c: {}'.format(c))
	return Vec3f(*c)


def test(image, vp):
	"""should be center"""
	pts = [
		(0, -0.5, 1),
		(0, 0.5, 1),
		(-0.5, 0, 1),
		(0.5, 0, 1),
		(0, 0, 1),
		(0.5, 0.5, 1)
	]
	vecs = [dot(vp, p) for p in pts]
	for i in range(0, len(vecs), 2):
		line(vecs[i], vecs[i + 1], image, red)


def draw_axis(image, vp):
	pts = [
		(0, 0, 1),
		(1, 0, 1),
		(0, 0, 1),
		(0, 1, 1),
	]
	colors = [red, green]
	vecs = map(Vec3i, [dot(vp, p) for p in pts])
	for i in range(0, len(vecs) // 2):
		line(vecs[i * 2], vecs[i * 2 + 1], image, colors[i])


def main():
	width, height = 100, 100
	image = Image.new('RGBA', (width, height), black)

	vertices, faces, _, _ = load_obj('../obj/cube.obj')

	w, h = width / 2., height / 2
	vp = viewport(w / 2., h / 2., w, h)

	# test(image, vp)
	draw_axis(image, vp)

	p = np.array([
		[1, 0, 0],
		[0, 1, 0],
		[-1./5, 0, 1]
	])
	vp = vp.dot(p)

	for face in faces[:1]:
		for i in range(len(face)):
			j = (i + 1) % len(face)
			v0 = vertices[face[i][0]]
			v1 = vertices[face[j][0]]

			sp0 = dot(vp, v0)
			sp1 = dot(vp, v1)
			
			if sp0.z and sp1.z:
				sp0 /= sp0.z
				sp1 /= sp1.z
				sp0 = Vec3i(sp0)
				sp1 = Vec3i(sp1)
				log.debug('sp: {}, {}'.format(sp0, sp1))
				line(sp0, sp1, image, white)

	image.transpose(Image.FLIP_TOP_BOTTOM).save('./output/perspective.png')

if __name__ == '__main__':
	main()

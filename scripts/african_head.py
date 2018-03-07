from common import *
from triangle import triangle, g_zbuffer
from model import Model
from random import randrange


g_light_dir = Vec3f(0, 0, -1)
g_gamma = 1.8


def flat_shading(iface, model, image):
	global g_light_dir
	global g_gamma

	def world2screen(v):
		x0 = (v.x + 1.) * (image.width - 1) // 2
		y0 = (v.y + 1.) * (image.height - 1) // 2  # avoid border error
		return Vec3f(x0, y0, v.z)

	screen_coords = []
	world_coords = []
	uv_coords = []
	for ivert in range(3):
		v = model.vert(iface, ivert)
		uv = model.uv(iface, ivert)

		uv_coords.append(uv)
		screen_coords.append(world2screen(v))
		world_coords.append(v)

	n = (world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0])
	n = n.normalize()
	intensity = (n * g_light_dir)

	# if intensity > 0:  # quick Back-face culling
	# 	intensity = intensity ** g_gamma
	# 	color = get_color(intensity)
	# 	triangle(screen_coords, image, color)

	if intensity > 0:
		triangle(screen_coords, image, uv_coords, model, intensity)


def main():
	global g_zbuffer
	width, height = 800, 800
	image = MyImage((width, height), 'RGBA')

	g_zbuffer.set(width, height)
	log.debug('g_zbuffer: {}'.format(g_zbuffer))
	model = Model('../obj/african_head.obj')

	for iface in range(model.nfaces()):
		flat_shading(iface, model, image)

	image.flip_vertically()
	image.save('./output/head.png')


if __name__ == '__main__':
	main()

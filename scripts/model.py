from common import *
from PIL import Image


class Model(object):

	def __init__(self, filename):
		self._verts, self._faces, self._uv, self._norms = self.load_obj(
			filename)
		self._diffusemap = self._load_texture(
			filename.replace('.obj', '_diffuse.tga'))

	@staticmethod
	def _load_texture(filename):
		return Image.open(filename).transpose(Image.FLIP_TOP_BOTTOM)

	@staticmethod
	def load_obj(filename):
		"""
		# below is data format:
				v 0.608654 -0.568839 -0.416318
				vt  0.429 0.548 0.000
				vn  -0.825 -0.117 0.552
				f 1193/1240/1193 1180/1227/1180 1179/1226/1179
		"""
		vertices = []
		faces = []
		textures = []
		normals = []
		# in wavefront obj file all indices start at 1, not zero
		normals.append(None)
		textures.append(None)
		vertices.append(None)
		# start parse .obj file
		with open(filename) as fp:
			for line in fp.readlines():
				mark = line[:2]
				fields = line.split()[1:]
				if mark == 'v ':
					vertices.append(Vec3f(*map(float, fields)))
				elif mark == 'f ':
					face = []
					for x in fields:
						# values = map(int, x.split('/'))
						# values = map(lambda x: x-1, values)
						# idx = int(x.split('/')[0])
						# face.append(idx)
						face.append(map(int, x.split('/')))
					faces.append(face)
				elif mark == 'vt':
					textures.append(Vec3f(*map(float, fields)))
				elif mark == 'vn':
					normals.append(Vec3f(*map(float, fields)))
		return vertices, faces, textures, normals

	def nfaces(self):
		return len(self._faces)

	def nverts(self):
		return len(self._verts)

	def face(self, idx):
		return [v[0] for v in self._faces[idx]]

	def vert(self, idx, ivert=None):
		if ivert is None:
			return self._verts[idx]
		return self._verts[self._faces[idx][ivert][0]]

	def diffuse(self, uv):
		return self._diffusemap.getpixel(tuple(uv))

	def uv(self, iface, ivert):
		idx = self._faces[iface][ivert][1]
		return Vec2i(self._uv[idx][0] * self._diffusemap.width, self._uv[idx][1] * self._diffusemap.height)

	def normal(self, iface, ivert):
		idx = self._faces[iface][ivert][2]
		return self._norms[idx].normalize()


def test():
	model = Model('../obj/african_head.obj')

	assert (model.nfaces(), model.nverts()) == (2492, 1259)

	iface = 10

	face = model.face(iface)

	for ivert in range(3):
		idx = face[ivert]

		v0 = model.vert(iface, ivert)
		v1 = model.vert(idx)

		assert v0 == v1

		uv = model.uv(iface, ivert)
		color = model.diffuse(uv)

	print (face, v0, v1, uv, color)
	print (model.normal(iface, ivert))

	# n1 = (model.vert(iface, 1) ^ model.vert(iface, 0)).normalize()
	# n2 = model.normal(iface, ivert)
	# assert n1 == n2, '{} == {}'.format(n1, n2)


if __name__ == '__main__':
	test()

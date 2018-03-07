import unittest
from operator import itemgetter


def optimize(cls):
	for idx, fld in enumerate(cls._fields):
		setattr(cls, fld, property(itemgetter(idx),
								   doc='Pointlias for field number {}'.format(idx)))
	return cls


# @optimize
class Point(object):

	_fields = ['x', 'y']

	def __init__(self, *arg):
		for k, v in zip(self._fields, map(int, arg)):
			setattr(self, k, v)

	def __eq__(self, other):
		return tuple(getattr(self, k) for k in self._fields) == other

	def __getitem__(self, index):
		return getattr(self, self._fields[index])


class TestSuite(unittest.TestCase):

	def test_namedtuple(self):
		a = Point(1, 2.1)
		b = Point(1.1, 2.3)
		self.assertEqual(a.x, 1)
		self.assertEqual(a.y, 2)
		self.assertEqual(a[0], 1)
		self.assertEqual(a[1], 2)
		self.assertEqual(a, b, (1, 2))


if __name__ == '__main__':
	unittest.main()

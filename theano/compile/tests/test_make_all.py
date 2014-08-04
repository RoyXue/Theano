"""
test of make_all method in vm.py
"""
import StringIO

import numpy

import theano
import theano.tensor as T

def test_make_all():
	x = [T.dvector(i) for i in ('a', 'b', 'c')]
	z = ((x[0] + x[1]) * x[2])
	# z = ((a + b) * c)
	test_mode = theano.Mode(linker='vm', optimizer='None')
	f = theano.function(x, z, mode=test_mode)
	print "ok"

if __name__ == "__main__":
	test_make_all()


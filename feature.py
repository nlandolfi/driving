import theano as th
import theano.tensor as tt
import numpy as np
import dynamics
import utils
import math

class Feature(object):
    def __init__(self, f):
        self.f = f
    def __call__(self, *args):
        return self.f(*args)
    def __add__(self, r):
        if hasattr(r, '__call__'):
            return Feature(lambda *args: self(*args)+r(*args))
        else:
            return Feature(lambda *args: self(*args)+r)
    def __radd__(self, r):
        if hasattr(r, '__call__'):
            return Feature(lambda *args: r(*args)+self(*args))
        else:
            return Feature(lambda *args: r+self(*args))
    def __mul__(self, r):
        return Feature(lambda *args: self(*args)*r)
    def __rmul__(self, r):
        return Feature(lambda *args: r*self(*args))
    def __pos__(self, r):
        return self
    def __neg__(self):
        return Feature(lambda *args: -self(*args))
    def __sub__(self, r):
        if hasattr(r, '__call__'):
            return Feature(lambda *args: self(*args)-r(*args))
        else:
            return Feature(lambda *args: self(*args)-r)
    def __rsub__(self, r):
        if hasattr(r, '__call__'):
            return Feature(lambda *args: r(*args)-self(*args))
        else:
            return Feature(lambda *args: r-self(*args))

def feature(f):
    return Feature(f)

def speed(s=1.):
    @feature
    def f(t, x, u):
        return -(x[3]-s)*(x[3]-s)
    return f

def control(bounds, width=0.05):
    @feature
    def f(t, x, u):
        ret = 0.
        for i, (a, b) in enumerate(bounds):
            if a is not None:
                ret += -tt.exp((a-u[i])/width)
            if b is not None:
                ret += -tt.exp((u[i]-b)/width)
        return ret
    return f

def deltaX(dyn):
    @feature
    def f(t, x, u):
        return dyn(x, u)[0] - x[0]
    return f

def deltaY(dyn):
    @feature
    def f(t, x, u):
        return dyn(x, u)[1] - x[1]
    return f

def deltaO(dyn):
    @feature
    def f(t, x, u):
        return dyn(x, u)[2] - x[2]
    return f

def deltaXN(dyn):
    @feature
    def f(t, x, u):
        delta = dyn(x, u)[:3] - x[:3]
        deltan = delta / tt.sqrt(tt.sum(tt.sqr(delta)))
        return deltan[0]
    return f

def deltaYN(dyn):
    @feature
    def f(t, x, u):
        delta = dyn(x, u)[:3] - x[:3]
        deltan = delta / tt.sqrt(tt.sum(tt.sqr(delta)))
        return deltan[1]
    return f

def deltaON(dyn):
    @feature
    def f(t, x, u):
        delta = dyn(x, u)[:3] - x[:3]
        deltan = delta / tt.sqrt(tt.sum(tt.sqr(delta)))
        return deltan[2]
    return f

if __name__ == '__main__':
    x = utils.vector(4)
    x.set_value([0, 0, math.pi/2, 0.5])
    u = utils.vector(2)
    u.set_value([0, .9])
    dyn = dynamics.CarDynamics(0.1)
    print(x.eval())
    print(dyn(x, u).eval())
    print(deltaX(dyn)(0, x, u).eval())
    print(deltaY(dyn)(0, x, u).eval())
    print(deltaO(dyn)(0, x, u).eval())

import pytest
import numpy as onp
from mindspore import Tensor, jit, jit_class


def match_array(actual, expected, error=0, err_msg=''):
    if isinstance(actual, (int, Tensor)):
        actual = actual.asnumpy() if isinstance(actual, Tensor) else onp.asarray(actual)

    if isinstance(expected, (int, tuple, Tensor)):
        expected = expected.asnumpy() if isinstance(
            expected, Tensor) else onp.asarray(expected)

    if error > 0:
        onp.testing.assert_almost_equal(
            actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


class StaticTestCall():
    def __init__(self):
        self.a = 2

    def __call__(self, x):
        return self.a * x


@jit_class
class MSTestCall():
    def __init__(self):
        self.a = 2

    def __call__(self, x):
        return self.a * x


class StaticTestAttribute():
    def __init__(self):
        self.a = 2
        self.b = 3.5


@jit_class
class MSTestAttribute():
    def __init__(self):
        self.a = 2
        self.b = 3.5


class StaticTestMethod():
    id = 0

    def __init__(self, a):
        self.a = a

    def func(self):
        self.id = self.id + self.a
        return self.id


@jit_class
class MSTestMethod():
    id = 0

    def __init__(self, a):
        self.a = a

    def func(self):
        self.id = self.id + self.a
        return self.id


@jit(mode="PIJit")
def call_class():
    net = StaticTestCall()
    res = net(2)
    return res


@jit(mode="PIJit")
def class_attribute():
    net = StaticTestAttribute()
    return net.a * net.b


@jit(mode="PIJit")
def class_attribute2():
    net = StaticTestAttribute()
    net.a = 3
    net.b = 2.5
    return net.a * net.b


@jit(mode="PIJit")
def class_method():
    net = StaticTestMethod(1)
    return net.func()


def ms_call_class():
    net = MSTestCall()
    res = net(2)
    return res


def ms_class_attribute():
    net = MSTestAttribute()
    return net.a * net.b


def ms_class_attribute2():
    net = MSTestAttribute()
    net.a = 3
    net.b = 2.5
    return net.a * net.b


def ms_class_method():
    net = MSTestMethod(1)
    return net.func()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [call_class])
@pytest.mark.parametrize('ms_func', [ms_call_class])
def test_parser_class1(func, ms_func):
    """
    Feature: Test __call__ method in class with PSJit and PIJit
    Description: Validate that the __call__ method works as expected in both static and JIT-optimized classes.
    Expectation: Both should return the same results.
    """
    result_static = func()
    result_ms = ms_func()
    match_array(result_static, result_ms)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [class_attribute])
@pytest.mark.parametrize('ms_func', [ms_class_attribute])
def test_parser_class2(func, ms_func):
    """
    Feature: Test class attributes in class with PSJit and PIJit
    Description: Validate that the attributes of the class work as expected in both static and JIT-optimized classes.
    Expectation: Both should return the same results.
    """
    result_static = func()
    result_ms = ms_func()
    match_array(result_static, result_ms)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [class_attribute2])
@pytest.mark.parametrize('ms_func', [ms_class_attribute2])
def test_parser_class3(func, ms_func):
    """
    Feature: Test modified class attributes in class with PSJit and PIJit
    Description: Validate that the modified attributes of the class work as expected
    in both static and JIT-optimized classes.
    Expectation: Both should return the same results.
    """
    result_static = func()
    result_ms = ms_func()
    match_array(result_static, result_ms)

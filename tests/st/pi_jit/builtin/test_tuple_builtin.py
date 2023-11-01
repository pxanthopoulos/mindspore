import pytest
import numpy as onp
from mindspore import Tensor, jit, context


def match_array(actual, expected, error=0, err_msg=''):
    if isinstance(actual, (int, tuple, list)):
        actual = onp.asarray(actual)

    if isinstance(expected, (int, tuple, list)):
        expected = onp.asarray(expected)

    if error > 0:
        onp.testing.assert_almost_equal(
            actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


@jit(mode="PIJit")
def fallback_tuple_with_input_list(a):
    res = tuple(a)
    return res


@jit(mode="PIJit")
def fallback_tuple_with_input_dict(a):
    res = tuple(a)
    return res


@jit(mode="PIJit")
def fallback_tuple_with_input_numpy_array(a):
    res = tuple(a)
    return res


@jit(mode="PIJit")
def fallback_tuple_with_input_numpy_tensor(a, b):
    res = tuple(a)
    res2 = tuple(b)
    res3 = tuple(())
    return res, res2, res3


@jit
def ms_fallback_tuple_with_input_list(a):
    res = tuple(a)
    return res


@jit
def ms_fallback_tuple_with_input_dict(a):
    res = tuple(a)
    return res


@jit
def ms_fallback_tuple_with_input_numpy_array():
    a = onp.array([1, 2, 3])
    res = tuple(a)
    return res


@jit
def ms_fallback_tuple_with_input_numpy_tensor(a, b):
    res = tuple(a)
    res2 = tuple(b)
    res3 = tuple(())
    return res, res2, res3


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [fallback_tuple_with_input_list])
@pytest.mark.parametrize('ms_func', [ms_fallback_tuple_with_input_list])
@pytest.mark.parametrize('a', [[1, 2, 3]])
def test_list_with_input_tuple(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    1. Test tuple() in PYNATIVE mode
    2. give the input data: list
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [fallback_tuple_with_input_dict])
@pytest.mark.parametrize('ms_func', [ms_fallback_tuple_with_input_dict])
@pytest.mark.parametrize('a', [{'a': 1, 'b': 2, 'c': 3}])
def test_list_with_input_dict(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    1. Test tuple() in PYNATIVE mode
    2. give the input data: dict
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [fallback_tuple_with_input_numpy_array])
@pytest.mark.parametrize('ms_func', [ms_fallback_tuple_with_input_numpy_array])
@pytest.mark.parametrize('a', [onp.array([1, 2, 3])])
def test_list_with_input_array(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for builtin list function support in PYNATIVE mode
    Expectation: the result match
    1. Test tuple() in PYNATIVE mode
    2. give the input data: numpy array
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [fallback_tuple_with_input_numpy_tensor])
@pytest.mark.parametrize('ms_func', [ms_fallback_tuple_with_input_numpy_tensor])
@pytest.mark.parametrize('a', [Tensor([1, 2])])
@pytest.mark.parametrize('b', [Tensor([1, 2]), Tensor([2, 3])])
def test_list_with_input_tensor(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for builtin list function support in PYNATIVE mode
    Expectation: the result match
    1. Test tuple() in PYNATIVE mode
    2. give the input data: tensor and (); output tuple
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res[0], ms_res[0], error=0, err_msg=str(ms_res))
    match_array(res[1], ms_res[1], error=0, err_msg=str(ms_res))
    match_array(res[2], ms_res[2], error=0, err_msg=str(ms_res))

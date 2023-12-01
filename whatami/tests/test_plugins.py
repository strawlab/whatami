# coding=utf-8
"""Test id string generation plugins on isolation."""
from collections import namedtuple, OrderedDict, defaultdict

PY2 = False
PY3 = True
from whatami import whatable

from whatami.plugins import (string_plugin, rng_plugin, has_numpy,
                             tuple_plugin, list_plugin, set_plugin, dict_plugin, numeric_type_plugin)
import pytest

# noinspection PyUnresolvedReferences
from .fixtures import df, array


def test_string_plugin():
    assert string_plugin("cuckoo's nest") == "'cuckoo\\'s nest'"
    assert string_plugin('') == "''"
    assert string_plugin(2) is None


def test_dict_plugin():

    # dict
    assert dict_plugin({}) == '{}'
    assert (dict_plugin({'int': 1, 'str': '1', 'func': test_dict_plugin}) ==
            "{'func':test_dict_plugin(),'int':1,'str':'1'}")

    # OrderedDict
    assert dict_plugin(OrderedDict()) == 'OrderedDict(seq=[])'
    assert (dict_plugin(OrderedDict([('int', 1), ('str', '1'), ('func', test_dict_plugin)])) ==
            "OrderedDict(seq=[('int',1),('str','1'),('func',test_dict_plugin())])")

    # defaultdict
    assert dict_plugin(defaultdict(int)) == 'defaultdict(default_factory=int(),seq={})'
    d = defaultdict(int)
    d[0] = 1
    assert dict_plugin(d) == 'defaultdict(default_factory=int(),seq={0:1})'


def test_set_plugin():

    # set
    assert set_plugin(set()) == 'set()'
    assert set_plugin({1, '1', test_set_plugin}) == "{'1',1,test_set_plugin()}"

    # frozenset
    assert set_plugin(frozenset()) == 'frozenset()'
    assert (set_plugin(frozenset([1, '1', test_set_plugin])) ==
            "frozenset({'1',1,test_set_plugin()})")

    # custom set
    class MySet(set):
        pass
    assert set_plugin(MySet()) == 'MySet(seq=set())'
    assert set_plugin(MySet([1, '1', test_set_plugin])) == "MySet(seq={'1',1,test_set_plugin()})"


def test_list_plugin():

    # Basic lists
    assert list_plugin([1, '1', test_list_plugin]) == "[1,'1',test_list_plugin()]"

    # Derived lists
    class MyList(list):
        pass
    assert list_plugin(MyList([1, '1', test_list_plugin])) == "MyList(seq=[1,'1',test_list_plugin()])"
    # maybe we should add a plugin specialised in named tuples, to include the names...


def test_tuple_plugin():

    # Basic tuples
    x = (1, '1', test_tuple_plugin)
    assert tuple_plugin(x) == "(1,'1',test_tuple_plugin())"

    # Subclasses
    x = namedtuple('namedt', ('x', 'y', 'z'))(1, '1', test_tuple_plugin)
    assert tuple_plugin(x) == "namedt(seq=(1,'1',test_tuple_plugin()))"
    # maybe we should add a plugin specialised in named tuples, to include the names...


@pytest.mark.skipif(not has_numpy(),
                    reason='the numpy RandomState plugin requires numpy')
def test_rng_plugin():
    # noinspection PyPackageRequirements
    import numpy as np
    # The state will depend on the seed...
    rng = np.random.RandomState(0)
    expected = "RandomState(state=dict(seq={'bit_generator':'MT19937','gauss':0.0,'has_gauss':0,'state':{'key':ndarray(hash='46423e261e128a1a0069fe54113b2ba2'),'pos':624}}))"
    assert expected == rng_plugin(rng)
    # ...and on where are we on the pseudo-random sampling chain
    rng.uniform(size=1)
    expected = "RandomState(state=dict(seq={'bit_generator':'MT19937','gauss':0.0,'has_gauss':0,'state':{'key':ndarray(hash='859ea260a4e83181802333a777b8bb95'),'pos':2}}))"
    assert expected == rng_plugin(rng)


def test_numpy_plugin(array):

    array, array_hash = array

    # joblib hash has changed?
    from whatami.plugins import hasher
    assert array_hash == hasher(array)

    @whatable
    def lpp(adjacency=array):  # pragma: no cover
        return adjacency

    assert lpp.what().id() == "lpp(adjacency=ndarray(hash='%s'))" % array_hash


def test_pandas_plugin(df):

    df, df_hash = df
    name = df.__class__.__name__

    # check for changes in joblib hashing and pandas pickling across versions
    from whatami.plugins import hasher
    assert df_hash == hasher(df)

    # check for proper string generation
    @whatable
    def lpp(adjacency=df):  # pragma: no cover
        return adjacency

    assert lpp.what().id() == "lpp(adjacency=%s(hash='%s'))" % (name, df_hash)


def test_numeric_type_plugin():
    assert numeric_type_plugin(int) == 'int()'
    assert numeric_type_plugin(float) == 'float()'
    assert numeric_type_plugin(complex) == 'complex()'

    @whatable
    def f(x=int):  # pragma: no cover
        return x
    assert f.what().id() == "f(x=int())"

# coding=utf-8
from ..what import whatable, whatareyou
from ..plugins import has_numpy, has_pandas, pd, np
import packaging.version
import pytest


@pytest.fixture
def c1():
    """A simple whatable object."""
    @whatable
    class C1(object):
        def __init__(self, p1='blah', p2='bleh', length=1):
            super(C1, self).__init__()
            self.p1 = p1
            self.p2 = p2
            self.length = length
            self._p1p2 = p1 + p2
            self.p2p1_ = p2 + p1
    return C1()


@pytest.fixture
def c2(c1):
    """A whatable object with a nested whatable."""
    @whatable
    class C2(object):
        def __init__(self, name='roxanne', c1=c1):
            super(C2, self).__init__()
            self.name = name
            self.c1 = c1
    return C2()


@pytest.fixture
def c3(c1, c2):
    """A whatable object with nested whatables and irrelevant members."""

    @whatable(force_flag_as_whatami=True)
    class C3(object):
        def __init__(self, c1=c1, c2=c2, irrelevant=True):
            super(C3, self).__init__()
            self.c1 = c1
            self.c2 = c2
            self.irrelevant = irrelevant

        def what(self):
            return whatareyou(self, non_id_keys=('irrelevant',))
    return C3()


def numpy_skip(test):  # pragma: no cover
    """Skips a test if the numpy plugin is not available."""
    if not has_numpy():
        return pytest.mark.skipif(test, reason='the numpy plugin requires numpy')
    return test


@pytest.fixture(params=list(map(numpy_skip, ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'])),
                ids=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'])
def array(request):
    """Hardcodes hashes, so we can detect hashing changes in joblib."""
    arrays = {
        # base array
        'a1': (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
               'a5546f2f3e43d4705e78b28a285cc10e'),
        # hash changes with dtype
        'a2': (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
               'aa2a05f2ab0bd6d00ed826a84d394fae'),
        'a3': (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.float64),
               '791245b6f3713a5f4893ab60fc4217c9'),
        # hash changes with shape and ndim
        'a4': (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).reshape((1, 9)),
               '51a4ecedef1ece8750a005c737bfd9da'),
        'a5': (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], ndmin=3),
               'f38021d73c5e2d076cdb88ec746c4d2a'),
        # hash changes with stride/order/contiguity
        'a6': (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], order='F'),
               'bdae6988a580b41846a7683d2e008408'),
        'a7': (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).T,
               'bdae6988a580b41846a7683d2e008408'),
    }
    return arrays[request.param]


def pandas_skip(test):  # pragma: no cover
    """Skips a test if the pandas plugin is not available."""
    # Check libraries are present
    if not has_pandas():
        return pytest.param(test, marks=pytest.mark.skip(reason='the pandas plugin requires pandas'))
    # Check library versions
    pandas_semver = "2.0"
    if packaging.version.parse(pd.__version__) < packaging.version.parse(pandas_semver):
        reason = 'these tests do not support pandas version %s' % pd.__version__
        return pytest.param(test, marks=pytest.mark.skip(reason=reason))
    return test


@pytest.fixture(params=list(map(pandas_skip, ['df1', 'df2', 'df3', 'df4', 's1', 's2'])),
                ids=['df1', 'df2', 'df3', 'df4', 's1', 's2'])
def df(request):
    """Hardcodes hashes, so we can detect hashing changes in joblib and pandas serialisation across versions."""
    # Unfortunate, that pandas hashes are unstable across pandas versions and python 2/3 should be documented
    adjacency = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    dfs = {
        'df1': (pd.DataFrame(data=adjacency, columns=['x', 'y', 'z']),
                '9c3bac426e867ee21b618145c564eba5'),
        'df2': (pd.DataFrame(data=adjacency, columns=['xx', 'yy', 'zz']),
                '417d435fb0e677a8eb13b41498440505'),
        'df3': (pd.DataFrame(data=adjacency.T, columns=['x', 'y', 'z']),
                'ff63ffde45da427c1d1588657ad70267'),
        'df4': (pd.DataFrame(data=adjacency, columns=['x', 'y', 'z'], index=['r1', 'r2', 'r3']),
                '8750de643c33fb1e049444b1d4362eb0'),
        's1': (pd.Series(data=adjacency.ravel()),
                'dfba7482fa680509b494c7da736c969c'),
        's2': (pd.Series(data=adjacency.ravel(), index=list(range(len(adjacency.ravel()))))[::-1],
                'ad4c81625f39a8ba183bf5cccc635312'),
    }
    return dfs[request.param]


@pytest.fixture(params=list(map(pandas_skip, ['dfw1'])), ids=['dfw1'])
def df_with_whatid(request):
    """Fixtures to test whatid manipulations mixed with pandas dataframes.

    Provides dataframes with:
      - A column "whatid" with the whatami ids
      - The rest of the columns must be the expectations for the extracted values, named as:
        - key: for top level keys
        - key1_key2_key3: for recursive keys
    """
    if request.param == 'dfw1':
        whatids = [
            "Blosc(cname='blosclz',level=5,shuffle=False)",
            "Blosc(cname='blosclz',level=6,shuffle=True)",
            "Blosc(cname='lz4hc',level=7,shuffle=True)",
            "Blosc(cname='lz4hc',level=8,shuffle=False)",
            "C2(c1=C1(length=1,p1='blah',p2='bleh'),name='roxanne')",
        ] * 4
        cnames = ['blosclz', 'blosclz', 'lz4hc', 'lz4hc', None] * 4
        levels = [5, 6, 7, 8, None] * 4
        shuffles = [False, True, True, False, None] * 4
        c1_lengths = [None, None, None, None, 1] * 4
        df = pd.DataFrame({'whatid': whatids,
                           'cname': cnames,
                           'level': levels,
                           'shuffle': shuffles,
                           'c1_length': c1_lengths})
        return df
    else:  # pragma: no cover
        raise ValueError('Unknown fixture %s' % request.param)

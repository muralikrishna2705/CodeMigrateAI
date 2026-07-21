# Python 2.7 to Python 3.12 — upgrade reference

Every construct below is either removed or behaviourally different in 3.12.

## Removed syntax

| Python 2.7 | Python 3.12 |
| --- | --- |
| `print x, y` | `print(x, y)` |
| `print >>sys.stderr, msg` | `print(msg, file=sys.stderr)` |
| `exec code in ns` | `exec(code, ns)` |
| `raise E, "msg", tb` | `raise E("msg").with_traceback(tb)` |
| `except E, e:` | `except E as e:` |
| `0777` | `0o777` |
| `10L` | `10` — `long` merged into `int` |
| `` `x` `` | `repr(x)` |
| `<>` | `!=` |
| `def f((a, b)):` | `def f(pair): a, b = pair` |
| `raw_input()` | `input()` |
| `input()` | `eval(input())` — but almost always a bug; keep `input()` |

## Integer division

The silent one. `/` was floor division for two ints in 2.7 and is true division
in 3.12, so it changes results without raising.

```python
# Python 2.7:  5 / 2 == 2
# Python 3.12: 5 / 2 == 2.5
average = total / count      # was integer, now float
midpoint = len(xs) / 2       # BREAKS: float index
```

Use `//` wherever the 2.7 code relied on truncation:

```python
midpoint = len(xs) // 2
```

## Text vs bytes

`str` in 2.7 was bytes; in 3.12 it is Unicode text and `bytes` is a distinct
type that does not implicitly convert.

| Python 2.7 | Python 3.12 |
| --- | --- |
| `unicode` | `str` |
| `str` (as binary) | `bytes` |
| `basestring` | `str` |
| `u"text"` | `"text"` |
| `unichr(n)` | `chr(n)` |
| `s.decode('utf-8')` | Only on `bytes` |
| `open(p)` for binary | `open(p, "rb")` |

```python
# 2.7 — worked because str was bytes
with open("data.bin") as f:
    header = f.read(4)

# 3.12
with open("data.bin", "rb") as f:
    header = f.read(4)
```

Always pass `encoding=` when opening text. In 2.7 the platform default was
usually harmless; in 3.12 it is a portability bug that surfaces as
`UnicodeDecodeError` on another machine.

```python
with open(path, encoding="utf-8") as f:
    ...
```

## Iterators, not lists

`dict.keys()`, `values()`, `items()`, `map`, `filter`, `zip`, and `range` all
return lazy views in 3.12.

| Python 2.7 | Python 3.12 |
| --- | --- |
| `d.iteritems()` | `d.items()` |
| `d.iterkeys()` | `d.keys()` |
| `d.has_key(k)` | `k in d` |
| `xrange(n)` | `range(n)` |
| `map(f, xs)` used as a list | `list(map(f, xs))` or `[f(x) for x in xs]` |
| `zip(a, b)` used twice | `list(zip(a, b))` |
| `reduce(f, xs)` | `functools.reduce(f, xs)` |

Wrap in `list()` **only** where the result is indexed, re-iterated, or mutated
while looping. Wrapping every call reintroduces the memory cost 3.x removed.

```python
# Mutating while iterating: the view would raise RuntimeError
for key in list(config.keys()):
    if key.startswith("_"):
        del config[key]
```

## Sorting

`cmp`, `cmp()`, and `sort(cmp=...)` are gone.

```python
# 2.7
items.sort(cmp=lambda a, b: cmp(a.priority, b.priority))

# 3.12
items.sort(key=lambda a: a.priority)
```

For a genuine multi-way comparison, use `functools.cmp_to_key`. Sorting mixed
types now raises `TypeError` instead of ordering by type name.

## Classes

All classes are new-style. `object` inheritance is implicit and should be
dropped.

```python
# 2.7
class Handler(Base, object):
    def __init__(self):
        super(Handler, self).__init__()

# 3.12
class Handler(Base):
    def __init__(self):
        super().__init__()
```

| Python 2.7 | Python 3.12 |
| --- | --- |
| `__nonzero__` | `__bool__` |
| `__div__` | `__truediv__` |
| `__cmp__` | `__eq__`/`__lt__` (+ `functools.total_ordering`) |
| `next()` method | `__next__()` |
| `im_func`, `func_name` | `__func__`, `__name__` |

## Standard library moves

| Python 2.7 | Python 3.12 |
| --- | --- |
| `ConfigParser` | `configparser` |
| `cPickle` | `pickle` |
| `StringIO.StringIO` | `io.StringIO` |
| `cStringIO` | `io.BytesIO` / `io.StringIO` |
| `urllib2`, `urlparse` | `urllib.request`, `urllib.parse` |
| `httplib` | `http.client` |
| `Queue` | `queue` |
| `SocketServer` | `socketserver` |
| `Tkinter` | `tkinter` |
| `thread` | `_thread` (prefer `threading`) |
| `commands` | `subprocess` |
| `os.getcwdu()` | `os.getcwd()` |

Removed in 3.12 specifically — a 2.7 codebase may still import these:

| Removed | Replacement |
| --- | --- |
| `imp` | `importlib` |
| `distutils` | `setuptools` / `packaging` |
| `asynchat`, `asyncore` | `asyncio` |
| `smtpd` | `aiosmtpd` |

## Comparison and equality

Ordering comparisons between unrelated types raise `TypeError`:

```python
# 2.7: sorted fine, ordered by type name
sorted([3, "a", None])
# 3.12: TypeError
```

`None` cannot be ordered at all, so `sort(key=...)` over a column that may be
`None` needs an explicit fallback:

```python
rows.sort(key=lambda r: (r.date is None, r.date))
```

## Type hints

3.12 supports the modern syntax; do not emit `typing.List`/`Dict`/`Optional`
for this target.

```python
def group(rows: list[Row], key: str) -> dict[str, list[Row]] | None:
    ...
```

3.12 also allows the compact generic form:

```python
def first[T](items: list[T]) -> T | None:
    return items[0] if items else None
```

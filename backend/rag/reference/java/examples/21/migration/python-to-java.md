# Python 3 to Java 21 — API and idiom mapping

Reference for converting Python source to Java. The hard direction: Python is
dynamically typed with duck typing, so every value needs a declared type and
several Python idioms have no direct Java form.

## Types

| Python | Java 21 |
| --- | --- |
| `int` (arbitrary precision) | `long`, or `BigInteger` when it can overflow |
| `float` | `double` |
| `str` | `String` |
| `bool` | `boolean` |
| `bytes` | `byte[]` |
| `list[T]` | `List<T>` (`ArrayList<>`) |
| `tuple` | `record`, or `List.of` when homogeneous |
| `dict[K, V]` | `Map<K, V>` (`HashMap<>`) |
| `set[T]` | `Set<T>` (`HashSet<>`) |
| `None` | `null`, or `Optional<T>` on a return |
| `Any` | `Object` |

**`int` is the trap.** Python integers are unbounded; Java `int` is 32-bit and
`long` is 64-bit, both wrapping silently on overflow. Factorials, hashes, ID
accumulators, and anything reading untrusted sizes need `BigInteger`. Choosing
`int` because the sample data is small is how this breaks in production.

`float` is not `Decimal`. If the Python used `decimal.Decimal` for money, the
Java is `BigDecimal`, never `double`.

## Collections

```python
nums = [1, 2, 3]
nums.append(4)
first = nums[0]
last = nums[-1]
chunk = nums[1:3]
```

```java
var nums = new ArrayList<>(List.of(1, 2, 3));
nums.add(4);
int first = nums.get(0);
int last = nums.getLast();          // Java 21
List<Integer> chunk = nums.subList(1, 3);
```

`subList` returns a **view**, not a copy: mutating it writes through to the
backing list, and structurally modifying the backing list invalidates it. Python
slicing copies. Wrap in `new ArrayList<>(...)` to match the Python semantics.

Negative indexing does not exist; `getFirst()`/`getLast()` arrived in 21.

```python
d = {"a": 1}
d["b"] = 2
v = d.get("c", 0)
for k, v in d.items(): ...
```

```java
var d = new HashMap<String, Integer>();
d.put("b", 2);
int v = d.getOrDefault("c", 0);
for (var e : d.entrySet()) { e.getKey(); e.getValue(); }
```

`HashMap` iteration order is unspecified; Python dicts preserve insertion order
since 3.7. If the Python relied on that — serialising, or producing stable
output — use `LinkedHashMap`.

## Comprehensions become streams

```python
evens = [x * 2 for x in nums if x % 2 == 0]
by_id = {u.id: u for u in users}
names = {u.name for u in users}
```

```java
var evens = nums.stream().filter(x -> x % 2 == 0).map(x -> x * 2).toList();
var byId  = users.stream().collect(Collectors.toMap(User::id, u -> u));
var names = users.stream().map(User::name).collect(Collectors.toSet());
```

`stream().toList()` returns an **immutable** list; `Collectors.toList()` returns
a mutable one. Python list comprehensions produce mutable lists, so use
`collect(Collectors.toList())` if the result is later mutated.

`Collectors.toMap` throws `IllegalStateException` on a duplicate key. A Python
dict comprehension silently keeps the last. To match, pass a merge function:
`Collectors.toMap(User::id, u -> u, (a, b) -> b)`.

A generator expression is lazy; `stream()` is too, so the mapping is faithful.
But a Java stream is single-use — consuming it twice throws
`IllegalStateException`.

## Strings

| Python | Java 21 |
| --- | --- |
| `f"{a} and {b}"` | `"%s and %s".formatted(a, b)` |
| `"-".join(parts)` | `String.join("-", parts)` |
| `s.split(",")` | `s.split(",")` — but the arg is a **regex** |
| `s.strip()` | `s.strip()` |
| `s.startswith(p)` | `s.startsWith(p)` |
| `s * 3` | `s.repeat(3)` |
| `len(s)` | `s.length()` |
| `"""..."""` | text block `"""..."""` (15+) |

`split` takes a regular expression in Java and a literal in Python. Splitting on
`"."` or `"|"` works in Python and silently splits on everything in Java — use
`Pattern.quote(".")`.

Python's `str.strip()` with no argument and Java's `String.strip()` both use a
Unicode definition of whitespace, so that one maps cleanly. `trim()` does not.

## None and Optional

```python
def find(uid: int) -> User | None:
    return db.get(uid)

u = find(1)
if u is not None:
    print(u.name)
```

```java
Optional<User> find(long uid) { return Optional.ofNullable(db.get(uid)); }

find(1).ifPresent(u -> System.out.println(u.name()));
```

Use `Optional` for return types only. As a field it is not serialisable and adds
an allocation; as a parameter it forces every caller to wrap. Java convention is
a nullable parameter with `@Nullable`, not `Optional`.

`Optional.of(null)` throws — `ofNullable` is the one that accepts null.

## Classes

```python
class Point:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y
    def __repr__(self): return f"Point({self.x}, {self.y})"
    def __eq__(self, o): return (self.x, self.y) == (o.x, o.y)
```

```java
public record Point(double x, double y) { }
```

A `record` supplies the constructor, accessors, `equals`, `hashCode`, and
`toString`. Accessors are `x()` and `y()`, not `getX()`.

`@dataclass` and `NamedTuple` both map to `record`. A mutable class with setters
does not — that stays a plain class.

```python
@dataclass
class Range:
    lo: int
    hi: int
    def __post_init__(self):
        if self.lo > self.hi: raise ValueError("lo > hi")
```

```java
public record Range(int lo, int hi) {
    public Range { if (lo > hi) throw new IllegalArgumentException("lo > hi"); }
}
```

## Exceptions

| Python | Java 21 |
| --- | --- |
| `raise ValueError(m)` | `throw new IllegalArgumentException(m)` |
| `raise KeyError(k)` | `throw new NoSuchElementException(k)` |
| `raise TypeError(m)` | `throw new ClassCastException(m)` |
| `except Exception as e:` | `catch (Exception e)` |
| `finally:` | `finally` |
| `with open(p) as f:` | `try (var f = Files.newBufferedReader(p))` |

Java distinguishes checked from unchecked exceptions; Python does not. A method
throwing a checked exception must declare `throws`, and every caller changes.
Prefer unchecked (`RuntimeException` subclasses) when porting, or the signature
change propagates through the whole call tree.

`with` maps to try-with-resources only when the object implements
`AutoCloseable`. A Python context manager doing something other than cleanup
(timing, locking, redirecting) has no Java equivalent — inline it or write a
lambda-taking helper.

## Files

```python
text = Path("f.txt").read_text(encoding="utf-8")
for line in open("f.txt"): ...
```

```java
String text = Files.readString(Path.of("f.txt"));          // UTF-8 default
try (var lines = Files.lines(Path.of("f.txt"))) { }        // must be closed
```

`Files.lines` returns a stream holding an open file handle; leaking it leaks the
descriptor. Python's file iterator is closed by the `with`.

## Truthiness

```python
if items: ...
if not name: ...
```

```java
if (!items.isEmpty()) { }
if (name == null || name.isEmpty()) { }
```

There is no truthiness in Java — only `boolean`. Every implicit Python truth
test becomes an explicit predicate, and `if (x)` on a `Integer` will not compile.
An empty string, empty list, `0`, and `None` are all falsy in Python and each
needs its own explicit check.

## No Java equivalent

| Python | Approach |
| --- | --- |
| `*args` / `**kwargs` | varargs `T...` for the first; a `Map<String, Object>` or builder for the second |
| multiple inheritance | interfaces with default methods |
| monkey patching | none — redesign |
| decorators | annotations + a framework proxy, or an explicit wrapper |
| `yield` generators | `Stream`, or an `Iterator` implementation |
| duck typing | declare an interface and implement it |
| tuple unpacking `a, b = f()` | return a `record` and destructure with a pattern |
| default mutable arg | none — Java evaluates defaults per call |

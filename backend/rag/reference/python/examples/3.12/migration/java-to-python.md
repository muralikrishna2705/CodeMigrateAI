# Java to Python 3.12 — API and idiom mapping

Reference for converting Java 8–21 source to idiomatic Python 3.12.

## Collections

| Java | Python 3.12 | Notes |
| --- | --- | --- |
| `List<T>` / `ArrayList<T>` | `list[T]` | `ArrayList<>()` → `[]` |
| `Map<K,V>` / `HashMap<>` | `dict[K, V]` | `new HashMap<>()` → `{}` |
| `Set<T>` / `HashSet<>` | `set[T]` | `new HashSet<>()` → `set()`, not `{}` |
| `LinkedList<T>` | `collections.deque` | Only when used as a queue |
| `Optional<T>` | `T | None` | Use `if x is not None`, not `.isPresent()` |
| `Arrays.asList(a, b)` | `[a, b]` | Java's is fixed-size; Python's is not |
| `Collections.unmodifiableList(x)` | `tuple(x)` | Python has no frozen list |
| `map.getOrDefault(k, d)` | `map.get(k, d)` | |
| `map.computeIfAbsent(k, f)` | `dict.setdefault(k, f())` | Or `collections.defaultdict` |
| `list.stream().filter(p).collect(toList())` | `[x for x in list if p(x)]` | Comprehension, not `filter()` |
| `list.stream().map(f).collect(toList())` | `[f(x) for x in list]` | |
| `stream().collect(joining(", "))` | `", ".join(parts)` | |

```java
Map<String, List<Order>> byCustomer = new HashMap<>();
for (Order o : orders) {
    byCustomer.computeIfAbsent(o.getCustomer(), k -> new ArrayList<>()).add(o);
}
```

```python
from collections import defaultdict

by_customer: defaultdict[str, list[Order]] = defaultdict(list)
for o in orders:
    by_customer[o.customer].append(o)
```

## Classes

Java getters/setters over private fields become plain public attributes. Adding
a getter "just in case" is not idiomatic — `@property` exists for when behaviour
is actually needed later, and adding it does not break callers.

```java
public class Point {
    private final int x, y;
    public Point(int x, int y) { this.x = x; this.y = y; }
    public int getX() { return x; }
    @Override public String toString() { return "Point(" + x + ", " + y + ")"; }
    @Override public boolean equals(Object o) { /* ... */ }
    @Override public int hashCode() { return Objects.hash(x, y); }
}
```

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: int
    y: int
```

`@dataclass` generates `__init__`, `__repr__`, `__eq__`; `frozen=True` adds
`__hash__` and makes it immutable, which is what `final` fields plus
`equals`/`hashCode` were expressing.

| Java | Python 3.12 |
| --- | --- |
| `interface` | `typing.Protocol`, or `abc.ABC` when inheritance is real |
| `abstract class` | `abc.ABC` + `@abstractmethod` |
| `enum Color { RED, GREEN }` | `class Color(enum.Enum): RED = auto()` |
| `static` method | `@staticmethod`, or a module-level function |
| `static` field | Class attribute |
| `record Point(int x, int y)` | `@dataclass(frozen=True)` |
| `toString()` | `__repr__` |
| `equals`/`hashCode` | `__eq__`/`__hash__` (both, or neither) |
| `Comparable.compareTo` | `__lt__` + `functools.total_ordering` |
| `Iterable.iterator()` | `__iter__` |
| `AutoCloseable.close()` | `__enter__`/`__exit__` |

## Exceptions

| Java | Python 3.12 |
| --- | --- |
| `IllegalArgumentException` | `ValueError` |
| `NullPointerException` | `AttributeError` / `TypeError` |
| `IndexOutOfBoundsException` | `IndexError` |
| `NoSuchElementException` | `KeyError` / `StopIteration` |
| `UnsupportedOperationException` | `NotImplementedError` |
| `IOException` | `OSError` |
| `NumberFormatException` | `ValueError` |
| `catch (A | B e)` | `except (A, B) as e:` |
| `finally` | `finally:`, or a `with` block |
| `throws` clause | Nothing — Python has no checked exceptions |

Do **not** invent a custom exception where a builtin fits. Python code raises
`ValueError`, not `IllegalArgumentException`.

## try-with-resources

```java
try (BufferedReader r = new BufferedReader(new FileReader(path))) {
    return r.readLine();
}
```

```python
with open(path, encoding="utf-8") as f:
    return f.readline()
```

Always pass `encoding=` to `open()`. Java's `FileReader` uses the platform
default charset, and reproducing that by omitting `encoding` reproduces a bug.

## Concurrency

| Java | Python 3.12 |
| --- | --- |
| `Thread` | `threading.Thread` |
| `ExecutorService` + `submit` | `concurrent.futures.ThreadPoolExecutor` |
| `Executors.newFixedThreadPool(n)` | `ThreadPoolExecutor(max_workers=n)` |
| `Future<T>.get()` | `concurrent.futures.Future.result()` |
| `CompletableFuture` | `asyncio` coroutines |
| `synchronized` method | `threading.Lock` held in a `with` block |
| `AtomicInteger` | A plain `int` guarded by a lock |
| `CountDownLatch` | `threading.Event` / `asyncio.Event` |

CPU-bound Java thread pools should become `ProcessPoolExecutor`, not
`ThreadPoolExecutor` — the GIL means threads do not give CPU parallelism.

```java
ExecutorService pool = Executors.newFixedThreadPool(4);
List<Future<Integer>> futures = new ArrayList<>();
for (String u : urls) futures.add(pool.submit(() -> fetch(u)));
for (Future<Integer> f : futures) total += f.get();
pool.shutdown();
```

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as pool:
    total = sum(pool.map(fetch, urls))
```

## Strings

| Java | Python 3.12 |
| --- | --- |
| `String.format("%s is %d", a, b)` | `f"{a} is {b}"` |
| `StringBuilder.append` in a loop | Build a `list`, then `"".join(parts)` |
| `str.isEmpty()` | `not s` |
| `str.equals(other)` | `s == other` |
| `str.equalsIgnoreCase(o)` | `s.casefold() == o.casefold()` |
| `str.split(",")` | `s.split(",")` — Java's takes a **regex**, Python's does not |
| `String.join(", ", list)` | `", ".join(list)` |
| `str.substring(a, b)` | `s[a:b]` |
| `str.trim()` | `s.strip()` |
| `str.chars()` | Iterate the string directly |

`String.split` taking a regex is a real trap: `"a.b".split(".")` in Java splits
on any character; in Python it splits on a literal dot. Use `re.split` only when
the Java pattern was genuinely a regex.

## Types and null

Java's `null` becomes `None`, and the check is `is None` — never `== None`.
Annotate optional values as `T | None` (3.10+ syntax; do not emit
`Optional[T]` for a 3.10+ target).

```python
def find_user(uid: int) -> User | None:
    return _users.get(uid)
```

## Entry point

```java
public static void main(String[] args) { new App().run(args); }
```

```python
def main(argv: list[str] | None = None) -> int:
    App().run(argv or sys.argv[1:])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

## Naming

`camelCase` → `snake_case` for methods, functions, variables, and fields.
`PascalCase` stays for classes. `CONSTANT_CASE` stays for module constants.
A leading underscore replaces `private`; Python has no access modifiers, and
inventing `__`-prefixed names to simulate them is wrong — name mangling is for
avoiding subclass collisions, not for privacy.

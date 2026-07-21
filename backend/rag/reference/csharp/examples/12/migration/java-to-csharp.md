# Java to C# 12 — API and idiom mapping

The closest pair in this corpus: both are statically typed, garbage-collected,
class-based, and run on a managed runtime. Most of the work is naming and
library, not restructuring. The differences that remain are the ones that bite.

## Naming and structure

| Java | C# |
| --- | --- |
| `camelCase` methods | `PascalCase` methods |
| `getName()` / `setName(v)` | a `Name { get; set; }` property |
| `package com.acme.util;` | `namespace Acme.Util;` |
| one public class per file | any number per file |
| `final` field | `readonly` field |
| `static final` constant | `const` (compile-time) or `static readonly` |
| interface `Runnable` | `Action` delegate |

`const` in C# is inlined into every calling assembly at compile time, so
changing one requires recompiling all callers. `static readonly` is resolved at
runtime. For anything crossing an assembly boundary, use `static readonly`.

## Types

| Java | C# |
| --- | --- |
| `int` / `long` | `int` / `long` |
| `boolean` | `bool` |
| `String` | `string` |
| `Integer` (boxed, nullable) | `int?` (`Nullable<int>`) |
| `Object` | `object` |
| `List<T>` | `List<T>` |
| `Map<K,V>` | `Dictionary<K,V>` |
| `Set<T>` | `HashSet<T>` |
| `T[]` | `T[]` |
| `Optional<T>` | `T?` with nullable reference types |
| `var` (10+) | `var` |

Java has no unsigned types; C# has `uint`, `ulong`, `byte`. A Java `int` holding
a value that should never be negative maps to `int` unless you are certain.

**Generics are not erased in C#.** `List<int>` stores unboxed ints and
`typeof(T)` works at runtime. Java tricks that exist to work around erasure —
passing `Class<T>` tokens, `@SuppressWarnings("unchecked")` casts,
`Array.newInstance` — are all unnecessary and should be deleted, not translated.

## Nullability

```java
public String find(long id) { return db.get(id); }   // may return null
```

```csharp
public string? Find(long id) => _db.GetValueOrDefault(id);
```

With `<Nullable>enable</Nullable>` (default in new projects), `string` means
non-null and `string?` means nullable, checked by the compiler. This is the
single biggest gain over Java and worth doing properly rather than suppressing
with `!`.

`?.`, `??`, and `??=` replace explicit null checks:

```csharp
var city = user?.Address?.City ?? "unknown";
_cache ??= new Dictionary<string, User>();
```

## Collections

```java
var list = new ArrayList<String>();
list.add("a");
String x = list.get(0);
int n = list.size();
var map = new HashMap<String, Integer>();
map.put("k", 1);
int v = map.getOrDefault("k", 0);
```

```csharp
var list = new List<string>();
list.Add("a");
string x = list[0];
int n = list.Count;
var map = new Dictionary<string, int>();
map["k"] = 1;
int v = map.GetValueOrDefault("k", 0);
```

`map.put(k, v)` and `map["k"] = v` both insert-or-update. But `map.get(k)` on a
missing key returns `null` in Java while `map["k"]` **throws**
`KeyNotFoundException` in C#. Use `TryGetValue` or `GetValueOrDefault`:

```csharp
if (map.TryGetValue(key, out var found)) { }
```

That difference is the most common runtime failure when porting Java map code.

## Streams become LINQ

| Java Stream | LINQ |
| --- | --- |
| `.stream().filter(p)` | `.Where(p)` |
| `.map(f)` | `.Select(f)` |
| `.flatMap(f)` | `.SelectMany(f)` |
| `.sorted(cmp)` | `.OrderBy(k)` / `.ThenBy(k)` |
| `.distinct()` | `.Distinct()` |
| `.limit(n)` / `.skip(n)` | `.Take(n)` / `.Skip(n)` |
| `.anyMatch(p)` | `.Any(p)` |
| `.allMatch(p)` | `.All(p)` |
| `.findFirst()` | `.FirstOrDefault()` |
| `.collect(toList())` | `.ToList()` |
| `.collect(toMap(k, v))` | `.ToDictionary(k, v)` |
| `.collect(groupingBy(k))` | `.GroupBy(k)` |
| `.reduce(id, op)` | `.Aggregate(id, op)` |
| `.count()` | `.Count()` |
| `IntStream.range(0, n)` | `Enumerable.Range(0, n)` |

```java
var names = users.stream()
    .filter(u -> u.age() >= 18)
    .map(User::name)
    .sorted()
    .toList();
```

```csharp
var names = users
    .Where(u => u.Age >= 18)
    .Select(u => u.Name)
    .Order()
    .ToList();
```

`.findFirst()` returns `Optional<T>`; `.FirstOrDefault()` returns `null` (or
`default(T)`) — so for a value type it returns `0`, not null. Use
`.Cast<T?>().FirstOrDefault()` or `.FirstOrDefault(x => …) is { } hit` when the
distinction matters.

LINQ is lazy like a stream, but a LINQ query **re-executes** on each enumeration
where a Java stream throws if consumed twice. A query over a database or a
mutating list gives different results each time it is iterated — call `.ToList()`
to materialise once.

## Records and classes

```java
public record Point(int x, int y) { }
```

```csharp
public record Point(int X, int Y);
```

Both give value equality, a constructor, and a `ToString`. C# records are
reference types by default; `record struct` gives a value type. C# adds
non-destructive mutation:

```csharp
var moved = point with { X = 10 };
```

Java has no `with` — you write a new constructor call.

## Exceptions

| Java | C# |
| --- | --- |
| `IllegalArgumentException` | `ArgumentException` |
| `IllegalStateException` | `InvalidOperationException` |
| `NullPointerException` | `NullReferenceException` |
| `IndexOutOfBoundsException` | `IndexOutOfRangeException` |
| `UnsupportedOperationException` | `NotSupportedException` |
| `IOException` | `IOException` |

**C# has no checked exceptions.** Every `throws IOException` clause disappears
and no caller is forced to handle anything. This removes compile errors, which
means it also removes the compiler's reminder that a failure path exists —
handling that Java forced on you is now yours to remember.

## Resources

```java
try (var reader = Files.newBufferedReader(path)) { }
```

```csharp
using var reader = new StreamReader(path);
```

`using` maps to try-with-resources and `IDisposable` to `AutoCloseable`. The
declaration form disposes at end of scope; the block form `using (…) { }` is
explicit.

## Async

Java 21 uses virtual threads with blocking calls; C# uses `async`/`await`.

```java
try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
    tasks.forEach(pool::submit);
}
```

```csharp
await Task.WhenAll(tasks.Select(ProcessAsync));
```

| Java | C# |
| --- | --- |
| `CompletableFuture<T>` | `Task<T>` |
| `.thenApply(f)` | `await` then use the value |
| `.get()` / `.join()` | `await` (never `.Result` — deadlocks) |
| `CompletableFuture.allOf` | `Task.WhenAll` |
| `ExecutorService` | `Task.Run` / the thread pool |

`async` is contagious in C# as it was not in Java 21: an `await` requires an
`async` method, which requires its caller to await, all the way up. A Java
method doing blocking I/O on a virtual thread has an ordinary signature; the C#
equivalent changes every signature on the call path. Budget for that — it is the
largest structural change in this migration.

Never call `.Result` or `.Wait()` to bridge sync and async. It deadlocks under a
synchronization context and is the classic C# async bug.

## Strings

| Java | C# |
| --- | --- |
| `"%s".formatted(x)` | `$"{x}"` (interpolation) |
| `String.join(",", parts)` | `string.Join(",", parts)` |
| `s.isBlank()` | `string.IsNullOrWhiteSpace(s)` |
| `s.equals(t)` | `s == t` — value comparison |
| `s.equalsIgnoreCase(t)` | `string.Equals(s, t, StringComparison.OrdinalIgnoreCase)` |
| `StringBuilder` | `StringBuilder` |
| text block `"""…"""` | raw string literal `"""…"""` |

`==` on strings compares **values** in C# and **references** in Java. Every
`s.equals(t)` becomes `s == t`, and — more important — any Java code that
correctly used `==` for reference identity must not be translated to `==`.

## Interfaces and inheritance

| Java | C# |
| --- | --- |
| `implements Foo` | `: IFoo` |
| `extends Base` | `: Base` |
| `@Override` | `override` (required, not advisory) |
| method overridable by default | `virtual` required on the base method |
| `abstract` | `abstract` |
| `sealed interface … permits` | `sealed` / `abstract` hierarchy |
| default method | default interface implementation (C# 8+) |

**C# methods are not virtual by default.** Marking a base method `virtual` and
the override `override` is mandatory; omitting them silently *hides* the base
method (`new`) instead of overriding it, so calls through a base-typed reference
run the base implementation. A Java override that is not marked in C# compiles
with a warning and behaves differently at runtime — the quietest failure in this
whole migration.

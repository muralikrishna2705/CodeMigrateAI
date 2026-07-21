# Java 8 to Java 21 — modernization reference

Behaviour-preserving upgrades available in Java 21 (LTS). Every feature below is
final in 21 unless noted.

## Local variable type inference (10+)

```java
Map<String, List<Order>> byCustomer = new HashMap<String, List<Order>>();
```

```java
var byCustomer = new HashMap<String, List<Order>>();
```

Only for locals with an initializer. Not for fields, parameters, or return
types. Do not use `var` where the initializer hides the type (`var x = foo()`).

## Records (16+)

Replace immutable data carriers — a final-fields class with a constructor,
getters, `equals`, `hashCode`, and `toString`.

```java
public final class Point {
    private final int x, y;
    public Point(int x, int y) { this.x = x; this.y = y; }
    public int getX() { return x; }
    public int getY() { return y; }
    @Override public boolean equals(Object o) { /* ... */ }
    @Override public int hashCode() { return Objects.hash(x, y); }
    @Override public String toString() { return "Point[x=" + x + ", y=" + y + "]"; }
}
```

```java
public record Point(int x, int y) { }
```

Accessors become `x()` and `y()`, **not** `getX()`. Every call site changes.
Only convert a class whose fields are all final and whose getters return them
unmodified — a getter with logic is not a record component.

Compact constructors keep validation:

```java
public record Range(int lo, int hi) {
    public Range {
        if (lo > hi) throw new IllegalArgumentException("lo > hi");
    }
}
```

## Switch expressions (14+)

```java
String label;
switch (day) {
    case SATURDAY:
    case SUNDAY:
        label = "weekend";
        break;
    default:
        label = "weekday";
}
```

```java
var label = switch (day) {
    case SATURDAY, SUNDAY -> "weekend";
    default -> "weekday";
};
```

Arrow form does not fall through, so `break` disappears. Use `yield` when a
branch needs a block. A switch **expression** over an enum must be exhaustive.

## Pattern matching for instanceof (16+)

```java
if (o instanceof String) {
    String s = (String) o;
    return s.length();
}
```

```java
if (o instanceof String s) {
    return s.length();
}
```

## Pattern matching for switch (21)

```java
return switch (shape) {
    case Circle c    -> Math.PI * c.radius() * c.radius();
    case Square s    -> s.side() * s.side();
    case null        -> 0;
    default          -> throw new IllegalArgumentException();
};
```

Record deconstruction, also final in 21:

```java
if (obj instanceof Point(int x, int y)) {
    return x + y;
}
```

## Sealed types (17+)

```java
public sealed interface Shape permits Circle, Square { }
```

Makes a `switch` over the permitted set exhaustive without a `default`, so
adding a subtype becomes a compile error rather than a silent fallthrough.

## Text blocks (15+)

```java
String q = "SELECT id, name\n" +
           "FROM users\n" +
           "WHERE active = true";
```

```java
String q = """
        SELECT id, name
        FROM users
        WHERE active = true""";
```

Incidental leading whitespace is stripped relative to the closing delimiter. A
trailing newline appears only if the closing `"""` is on its own line — moving
it changes the string.

## Collections

| Java 8 | Java 21 |
| --- | --- |
| `Arrays.asList(a, b)` | `List.of(a, b)` — immutable |
| `new ArrayList<>(Arrays.asList(...))` | `new ArrayList<>(List.of(...))` |
| `Collections.unmodifiableList(x)` | `List.copyOf(x)` |
| `stream().collect(Collectors.toList())` | `stream().toList()` — immutable |
| `map.get(k)` + null check | `map.getOrDefault(k, d)` |
| `list.get(list.size() - 1)` | `list.getLast()` (21) |
| `list.get(0)` | `list.getFirst()` (21) |

`List.of` and `stream().toList()` return **immutable** lists. Code that later
mutates the result gets `UnsupportedOperationException` at runtime, not a
compile error — check each call site before converting.

`List.of` also rejects nulls. `Arrays.asList(a, null)` is legal; `List.of(a,
null)` throws `NullPointerException`.

## Streams

| Java 8 | Java 21 |
| --- | --- |
| `.filter(...).findFirst().isPresent()` | `.anyMatch(...)` |
| `.collect(Collectors.toList())` | `.toList()` |
| `IntStream.range(0,n).mapToObj(...)` | Often a plain enhanced for |
| — | `.mapMulti(...)` (16+) |

`Stream.toList()` allows nulls; `Collectors.toUnmodifiableList()` does not.

## Optional

| Java 8 | Java 21 |
| --- | --- |
| `if (o.isPresent()) { ... }` | `o.ifPresent(v -> ...)` |
| `o.isPresent() ? o.get() : d` | `o.orElse(d)` |
| — | `o.ifPresentOrElse(f, r)` (9+) |
| — | `o.or(() -> other)` (9+) |
| — | `o.stream()` (9+) |
| `!o.isPresent()` | `o.isEmpty()` (11+) |

## Concurrency

Virtual threads (21) suit blocking I/O workloads. Do **not** convert a
CPU-bound pool — virtual threads give no CPU parallelism.

```java
ExecutorService pool = Executors.newFixedThreadPool(200);   // I/O-bound
```

```java
try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
    tasks.forEach(pool::submit);
}
```

`ExecutorService` extends `AutoCloseable` in 19+, so try-with-resources replaces
`shutdown()` / `awaitTermination()`.

Virtual threads make thread-pool sizing obsolete but **pin** on `synchronized`
blocks that block. Prefer `ReentrantLock` in code moving to virtual threads.

## Removed and deprecated

| Java 8 | Java 21 |
| --- | --- |
| `new Integer(5)` | `Integer.valueOf(5)` — constructors removed |
| `new Double(1.0)` | `Double.valueOf(1.0)` |
| `Class.newInstance()` | `getDeclaredConstructor().newInstance()` |
| `finalize()` | `java.lang.ref.Cleaner` |
| `SecurityManager` | Removed — no replacement |
| `Thread.stop()` | Removed — throws `UnsupportedOperationException` |
| `Runtime.exec(String)` | `Runtime.exec(String[])` |
| `java.util.Date` | `java.time.Instant` / `LocalDateTime` |
| `SimpleDateFormat` | `DateTimeFormatter` — thread-safe |
| `Calendar` | `java.time.LocalDate` etc. |

Boxed constructors are removal-flagged and produce a warning in 21.

## String

| Java 8 | Java 21 |
| --- | --- |
| `str.trim()` | `str.strip()` — Unicode-aware |
| manual blank check | `str.isBlank()` (11+) |
| `Collections.nCopies` + join | `str.repeat(n)` (11+) |
| manual line splitting | `str.lines()` (11+) |
| `String.format(...)` | `str.formatted(...)` (15+) |

`trim()` removes chars `<= U+0020`; `strip()` uses `Character.isWhitespace`.
They differ on Unicode whitespace, so this is not a pure rename.

## Files

| Java 8 | Java 21 |
| --- | --- |
| `new BufferedReader(new FileReader(p))` | `Files.newBufferedReader(path)` |
| manual read loop | `Files.readString(path)` (11+) |
| manual write | `Files.writeString(path, s)` (11+) |

`FileReader` uses the platform default charset; `Files.readString` defaults to
UTF-8. If the original relied on the platform default, pass the charset
explicitly rather than silently changing the encoding.

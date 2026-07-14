# Java Modernization Idioms

Reference patterns for migrating to modern Java (11+ / 17+).

## Use var for local type inference (Java 10+)

```java
var names = new ArrayList<String>();
var total = computeTotal(orders);
```

## Records for immutable data carriers (Java 16+)

```java
public record Point(double x, double y) {}
```

## Enhanced switch expressions (Java 14+)

```java
String label = switch (status) {
    case ACTIVE -> "running";
    case PAUSED -> "waiting";
    default -> "unknown";
};
```

## Streams over manual loops

```java
List<String> upper = names.stream()
    .filter(Objects::nonNull)
    .map(String::toUpperCase)
    .collect(Collectors.toList());
```

## Text blocks for multi-line strings (Java 15+)

```java
String json = """
    {
      "name": "example"
    }
    """;
```

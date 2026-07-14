# Python Modernization Idioms

Reference patterns for migrating to modern Python (3.8+).

## Prefer f-strings over % and .format()

```python
# Legacy
name = "world"
greeting = "hello %s" % name

# Modern
greeting = f"hello {name}"
```

## Use pathlib instead of os.path

```python
from pathlib import Path

config = Path(__file__).resolve().parent / "config.toml"
if config.exists():
    text = config.read_text(encoding="utf-8")
```

## Dataclasses for plain data containers

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
```

## Context managers for resource handling

```python
with open("data.txt", encoding="utf-8") as f:
    for line in f:
        process(line)
```

## Type hints on public functions

```python
def total(values: list[int]) -> int:
    return sum(values)
```

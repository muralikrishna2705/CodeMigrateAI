# JavaScript to TypeScript 5.x — migration reference

Converting untyped JavaScript to TypeScript. Types are erased at compile time:
they constrain the compiler, never the runtime. Nothing below changes behaviour
unless noted.

## Project setup

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "dist"
  },
  "include": ["src/**/*"]
}
```

`strict: true` is the setting that makes the migration worth doing — it turns on
`strictNullChecks`, `noImplicitAny`, and five others. On a large codebase, start
with `strict: false` plus `allowJs: true`, convert file by file, then turn it on.
Converting everything at once under `strict` produces thousands of errors at
which point nobody can tell the real ones from the noise.

## Variables and inference

```javascript
var count = 0;
let name = "widget";
const items = [];
```

```typescript
let count = 0;                    // inferred number
const name = "widget";            // inferred "widget" (literal type)
const items: string[] = [];       // MUST annotate: [] infers never[]
```

Annotate only where inference fails or the type is the API contract. Writing
`const name: string = "widget"` is noise; `const items: string[] = []` is
necessary, because an empty array literal infers `never[]` and every later
`push` is an error.

## Functions

```javascript
function area(w, h) { return w * h; }
const scale = (v, by) => v * by;
function greet(name, greeting) {
  greeting = greeting || "Hello";
  return greeting + ", " + name;
}
```

```typescript
function area(w: number, h: number): number { return w * h; }
const scale = (v: number, by: number): number => v * by;
function greet(name: string, greeting = "Hello"): string {
  return `${greeting}, ${name}`;
}
```

Parameters need annotations (`noImplicitAny`); return types are inferred, but
annotate exported functions so a change to the body cannot silently change the
public type.

`greeting || "Hello"` and `greeting = "Hello"` are **not** equivalent: the
default applies only to `undefined`, while `||` also replaces `""` and `0`. Use
`??` when you mean "null or undefined only".

## `any` vs `unknown`

```typescript
function parse(raw: any) { return raw.data.items; }      // no checking at all
function parse(raw: unknown) {
  if (typeof raw === "object" && raw !== null && "data" in raw) { /* ... */ }
  throw new TypeError("unexpected payload");
}
```

`any` disables checking and spreads: every expression derived from it is also
`any`. Prefer `unknown` at boundaries (JSON, `catch`, third-party) and narrow.
`JSON.parse` returns `any` — assign it to an `unknown` immediately.

In `catch`, the binding is `unknown` under `useUnknownInCatchVariables`
(implied by `strict`). `err.message` is an error; narrow with
`err instanceof Error` first.

## Objects: interface or type alias

```javascript
const user = { id: 1, name: "ada", email: null };
```

```typescript
interface User {
  id: number;
  name: string;
  email: string | null;   // present but nullable
  nickname?: string;      // may be absent
}
```

`email: string | null` and `nickname?: string` are different contracts. Optional
means the key may be missing; a nullable union means the key is there and may
hold null. Mapping a JSON schema, `required` decides which you want.

Use `interface` for object shapes that may be extended or implemented; `type`
for unions, tuples, and mapped types. `type` cannot be reopened by declaration
merging, which is usually what you want.

## Union types replace string flags

```javascript
function move(direction) { /* "up" | "down" | anything */ }
```

```typescript
type Direction = "up" | "down" | "left" | "right";
function move(direction: Direction) { }
```

The single highest-value conversion in most codebases: a typo in a string flag
becomes a compile error instead of a silent no-op branch.

Discriminated unions replace `if (obj.kind === ...)` chains:

```typescript
type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "square"; side: number };

function area(s: Shape): number {
  switch (s.kind) {
    case "circle": return Math.PI * s.radius ** 2;
    case "square": return s.side ** 2;
  }
}
```

The compiler narrows `s` inside each branch, and a new variant makes the
function fail to compile if the return type is annotated.

## Classes

```javascript
class Repo {
  constructor(db) {
    this.db = db;
    this.cache = new Map();
  }
}
```

```typescript
class Repo {
  private readonly cache = new Map<string, User>();
  constructor(private readonly db: Database) {}
}
```

Parameter properties (`private readonly db: Database` in the constructor
signature) declare and assign in one step.

`private` is compile-time only — it is erased and the field is reachable at
runtime. Use `#field` for real runtime privacy. `readonly` likewise prevents
reassignment only in type-checked code.

Under `strictPropertyInitialization`, a field with no initializer and no
constructor assignment is an error. Use `!` only when something outside the
constructor guarantees assignment (a framework lifecycle hook), never to
silence the check.

## Modules

| JavaScript (CJS) | TypeScript (ESM) |
| --- | --- |
| `const fs = require("fs")` | `import * as fs from "fs"` |
| `const { join } = require("path")` | `import { join } from "path"` |
| `module.exports = Foo` | `export default Foo` |
| `module.exports = { a, b }` | `export { a, b }` |
| `exports.a = a` | `export const a = ...` |

`esModuleInterop: true` is what makes `import express from "express"` work
against a CommonJS default export. Without it you need
`import express = require("express")`.

Use `import type { User } from "./types"` for type-only imports so the bundler
can drop them; under `verbatimModuleSyntax` this is required, not optional.

## Async

```javascript
function load(id, cb) {
  fetch("/u/" + id).then(r => r.json()).then(d => cb(null, d)).catch(cb);
}
```

```typescript
async function load(id: string): Promise<User> {
  const res = await fetch(`/u/${id}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as User;
}
```

`res.json()` returns `Promise<any>`. The `as User` is an **unchecked** claim —
it does not validate. If the payload can be wrong, parse with a runtime
validator (zod, io-ts) and derive the type from the schema.

An `async` function always returns a promise, so `Promise<User>` is the
annotation, not `User`.

## Generics

```javascript
function first(arr) { return arr[0]; }
```

```typescript
function first<T>(arr: readonly T[]): T | undefined { return arr[0]; }
```

`T | undefined` is the honest return type — an empty array yields `undefined`.
Under `noUncheckedIndexedAccess` the compiler enforces this; without that flag
`arr[0]` types as `T` and lies.

## Type assertions and narrowing

```typescript
const el = document.getElementById("app") as HTMLCanvasElement;  // unchecked
const el = document.getElementById("app");
if (el instanceof HTMLCanvasElement) { /* narrowed, verified */ }
```

`as` tells the compiler to stop arguing; it inserts no check. Every `as` in a
migration is a place where a runtime error can still happen. Prefer `instanceof`,
`typeof`, `in`, or a user-defined type guard:

```typescript
function isUser(v: unknown): v is User {
  return typeof v === "object" && v !== null && "id" in v;
}
```

## Untyped dependencies

```typescript
// 1. Try the community types first:
//    npm i -D @types/lodash
// 2. Otherwise declare a local shim in src/types/legacy.d.ts:
declare module "legacy-widget" {
  export function render(el: HTMLElement, opts?: Record<string, unknown>): void;
}
```

## Things that change behaviour

Almost nothing here does, with these exceptions:

| Construct | Risk |
| --- | --- |
| `enum` | Emits a real runtime object. Prefer `const enum` (erased) or a union of literals. |
| `class` field initializers | Ordering differs from an assignment in the constructor body. |
| decorators | Emit runtime code; semantics differ between the legacy and standard proposals. |
| `import` → `require` interop | A default import of a CJS module resolves differently under `esModuleInterop`. |

Everything else — annotations, interfaces, generics, `as` — is erased. If a
migrated file behaves differently and none of the above appear in it, the cause
is a genuine code edit made during conversion, not the types.

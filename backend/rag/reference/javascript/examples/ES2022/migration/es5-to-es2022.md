# ES5 to ES2022 — JavaScript modernization reference

In-language upgrade. Every construct below is supported in Node 18+ and all
current browsers. Behaviour-preserving unless noted.

## var, let, const

```javascript
var total = 0;
for (var i = 0; i < items.length; i++) {
  var item = items[i];
  total += item.price;
}
```

```javascript
let total = 0;
for (const item of items) {
  total += item.price;
}
```

`var` is function-scoped and hoisted; `let`/`const` are block-scoped. This is
**not** always behaviour-preserving — the classic case:

```javascript
for (var i = 0; i < 3; i++) setTimeout(() => console.log(i));  // 3 3 3
for (let i = 0; i < 3; i++) setTimeout(() => console.log(i));  // 0 1 2
```

Code that relied on `var` leaking out of a block changes meaning. Convert to
`const` first and let the reassignment errors tell you which need `let`.

## Arrow functions

```javascript
var doubled = nums.map(function (n) { return n * 2; });
var self = this;
button.onclick = function () { self.handle(); };
```

```javascript
const doubled = nums.map((n) => n * 2);
button.onclick = () => this.handle();
```

Arrows have no own `this` — they close over the enclosing scope, which is
exactly what the `var self = this` idiom was working around.

Do **not** convert:
- object methods that use `this` (`{ greet: function () { return this.name; } }`)
- constructors — arrows cannot be called with `new`
- anything using `arguments` — arrows have none; use rest `(...args)`
- prototype methods assigned to `Foo.prototype.bar`

## Template literals

```javascript
var msg = "Hello, " + name + "! You have " + count + " items.";
var sql = "SELECT id\n" + "FROM users\n" + "WHERE active = 1";
```

```javascript
const msg = `Hello, ${name}! You have ${count} items.`;
const sql = `SELECT id
FROM users
WHERE active = 1`;
```

A template literal keeps every newline and all indentation, so a multi-line
template inside an indented block embeds that indentation in the string.

## Destructuring

```javascript
var id = user.id;
var name = user.name;
var first = arr[0], second = arr[1];

function draw(opts) {
  var color = opts.color || "black";
  var width = opts.width || 1;
}
```

```javascript
const { id, name } = user;
const [first, second] = arr;

function draw({ color = "black", width = 1 } = {}) { }
```

The `= {}` on the parameter matters: without it, calling `draw()` with no
argument throws when destructuring `undefined`.

A destructuring default fires only on `undefined`, while `||` also replaces
`""`, `0`, and `false`. Where the old code used `||` on a value that can
legitimately be `0`, the ES5 was already buggy and the conversion fixes it —
verify that is what you want.

## Spread and rest

| ES5 | ES2022 |
| --- | --- |
| `arr.slice()` | `[...arr]` |
| `a.concat(b)` | `[...a, ...b]` |
| `Math.max.apply(null, arr)` | `Math.max(...arr)` |
| `Array.prototype.slice.call(arguments)` | `(...args) => {}` |
| `$.extend({}, defaults, opts)` | `{ ...defaults, ...opts }` |
| `f.apply(ctx, args)` | `f.call(ctx, ...args)` |

Spread is a **shallow** copy. `{ ...config }` shares every nested object with
the original, so mutating `copy.db.host` mutates the source. Use
`structuredClone(config)` for a deep copy.

## Object shorthand and computed keys

```javascript
var point = { x: x, y: y, toString: function () { return "p"; } };
var obj = {};
obj[key] = value;
```

```javascript
const point = { x, y, toString() { return "p"; } };
const obj = { [key]: value };
```

## Classes

```javascript
function Animal(name) { this.name = name; }
Animal.prototype.speak = function () { return this.name + " makes a sound"; };

function Dog(name) { Animal.call(this, name); }
Dog.prototype = Object.create(Animal.prototype);
Dog.prototype.constructor = Dog;
```

```javascript
class Animal {
  #id = crypto.randomUUID();          // truly private (ES2022)
  static kingdom = "Animalia";        // static field (ES2022)
  constructor(name) { this.name = name; }
  speak() { return `${this.name} makes a sound`; }
}

class Dog extends Animal {
  speak() { return `${this.name} barks`; }
}
```

Class bodies are always strict mode, and class declarations are **not** hoisted
the way function declarations are — calling `new Dog()` above its declaration
throws `ReferenceError` where the ES5 prototype version worked.

`#private` fields are enforced at runtime; accessing one from outside is a
syntax error, not `undefined`. The old `_private` convention was advisory.

## Async

```javascript
getUser(id, function (err, user) {
  if (err) return done(err);
  getOrders(user.id, function (err, orders) {
    if (err) return done(err);
    done(null, { user: user, orders: orders });
  });
});
```

```javascript
async function load(id) {
  const user = await getUser(id);
  const orders = await getOrders(user.id);
  return { user, orders };
}
```

Independent awaits should not be sequential:

```javascript
const [user, config] = await Promise.all([getUser(id), getConfig()]);
```

`await` in a loop serialises the whole loop — the most common performance
regression when converting callbacks. Use `Promise.all` over a `map` when the
iterations are independent.

`Promise.all` rejects on the first failure; `Promise.allSettled` (ES2020) waits
for all and reports each outcome.

Top-level `await` works in ES modules (ES2022), not in CommonJS.

## Modules

| ES5 | ES2022 |
| --- | --- |
| IIFE namespace | `export` / `import` |
| `require("x")` | `import x from "x"` |
| `module.exports = f` | `export default f` |
| `exports.a = a` | `export const a = ...` |

Modules are strict mode and deferred by default. A script relying on implicit
globals (`x = 1` with no declaration) throws once converted.

## Array and object methods

| ES5 | Modern |
| --- | --- |
| `arr.indexOf(x) !== -1` | `arr.includes(x)` |
| manual loop to find | `arr.find(fn)` / `arr.findIndex(fn)` |
| `arr.filter(fn)[0]` | `arr.find(fn)` |
| nested `concat` flatten | `arr.flat()` / `arr.flatMap(fn)` |
| `arr[arr.length - 1]` | `arr.at(-1)` (ES2022) |
| `Object.keys(o).forEach` | `Object.entries(o).forEach(([k, v]) => …)` |
| `hasOwnProperty.call(o, k)` | `Object.hasOwn(o, k)` (ES2022) |
| pairs → object, manual | `Object.fromEntries(pairs)` |

`includes` finds `NaN` where `indexOf` never does — `[NaN].indexOf(NaN)` is
`-1`, `[NaN].includes(NaN)` is `true`.

## Optional chaining and nullish coalescing

```javascript
var city = user && user.address && user.address.city;
var port = config.port || 8080;
```

```javascript
const city = user?.address?.city;
const port = config.port ?? 8080;
```

`??` falls back only on `null`/`undefined`; `||` also falls back on `0`, `""`,
and `false`. `config.port || 8080` on a configured port of `0` gives 8080 —
converting to `??` changes that, and the new behaviour is almost always the
intended one.

`?.()` and `?.[]` cover calls and index access:
`callbacks.onDone?.()`, `matrix?.[row]?.[col]`.

## Other useful upgrades

| ES5 | Modern |
| --- | --- |
| `parseInt(s, 10)` for validation | `Number.parseInt` / `Number.isInteger` |
| `isNaN(v)` | `Number.isNaN(v)` — no coercion |
| object as a keyed store | `Map` (any key type, ordered, `.size`) |
| object as a set | `Set` |
| `for...in` over an array | `for...of`, or `.entries()` for the index |
| `new Date()` arithmetic | `Intl.DateTimeFormat`, or a date library |
| `try { } catch (e) { }` unused binding | `catch { }` (ES2019) |

`isNaN("abc")` is `true` because it coerces first; `Number.isNaN("abc")` is
`false`. They are different functions, not aliases — this is a behaviour change
and usually a bug fix.

`for...in` iterates inherited enumerable keys and gives string indices on an
array. It was never correct for arrays; `for...of` is.

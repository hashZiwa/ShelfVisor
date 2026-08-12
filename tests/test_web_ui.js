const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

test("final call-number label omits the spine index", () => {
  const element = {
    addEventListener() {},
    classList: { add() {}, remove() {}, toggle() {} },
  };
  const context = vm.createContext({
    document: {
      body: element,
      querySelector: () => element,
      querySelectorAll: () => [],
    },
    fixture: {
      call_number: "811.3 A",
      expected_rank: 7,
      index: 7,
      status: "ok",
    },
  });
  const appScript = readFileSync(path.join(__dirname, "..", "web", "app.js"), "utf8");

  vm.runInContext(`${appScript}\nglobalThis.renderedSpine = renderSpine(globalThis.fixture);`, context);

  assert.match(context.renderedSpine, /<span class="label">811\.3 A<\/span>/);
  assert.doesNotMatch(context.renderedSpine, /<span class="label">7\.\s/);
});

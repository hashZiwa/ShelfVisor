# Call-Number List Without Indexes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the numeric spine index prefix from call numbers shown in the browser's final results list.

**Architecture:** Preserve the analysis payload and image-rendering pipeline. Change only the browser template responsible for a final result row, protected by a focused Node test that executes the real `renderSpine` function in a minimal browser-like VM context.

**Tech Stack:** Vanilla JavaScript, Node.js built-in test runner and VM

## Global Constraints

- The final result label changes from `1. 811.3 A` to `811.3 A`.
- The analysis payload's `index` field remains unchanged.
- Generated result and debug image labels remain unchanged.
- Batch and debug navigation numbering remains unchanged.

---

### Task 1: Remove the Final-List Index Prefix

**Files:**
- Create: `tests/test_web_ui.js`
- Modify: `web/app.js:396-406`

**Interfaces:**
- Consumes: `renderSpine(spine)` with `spine.call_number`, `spine.index`, `spine.status`, and `spine.expected_rank`.
- Produces: The existing HTML string from `renderSpine(spine)`, with the `.label` text containing only `${spine.call_number}`.

- [ ] **Step 1: Write the failing test**

```javascript
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
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `node --test tests/test_web_ui.js`

Expected: FAIL because `web/app.js` still contains `${spine.index}. ${spine.call_number}`.

- [ ] **Step 3: Write the minimal implementation**

Replace the final-list label in `renderSpine` with:

```javascript
<span class="label">${spine.call_number}</span>
```

- [ ] **Step 4: Run the focused and full test suites**

Run: `node --test tests/test_web_ui.js`

Expected: PASS.

Run: `python -m unittest discover -s tests -v`

Expected: All tests PASS with no errors or failures.

- [ ] **Step 5: Verify the diff is scoped and commit**

Run: `git diff --check`

Expected: No output and exit code 0.

```bash
git add tests/test_web_ui.js web/app.js docs/superpowers/plans/2026-08-12-call-number-list-without-index.md
git commit -m "fix: remove indexes from call number list"
```

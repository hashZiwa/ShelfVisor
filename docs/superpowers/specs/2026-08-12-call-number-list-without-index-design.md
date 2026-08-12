# Remove Indexes from the Final Call-Number List

## Goal

Show each call number in the browser's final results list without the internal spine index prefix.

## Scope

- Change the final result label from `1. 811.3 A` to `811.3 A`.
- Keep the `index` field in analysis data unchanged.
- Keep numbered labels on generated result and debug images unchanged.
- Keep numbering in batch and debug navigation unchanged.

## Implementation

Update the result-list template in `web/app.js` so its label contains only the escaped call-number value. No CSS or server-side data transformation is needed.

## Verification

Add a focused rendering test that confirms the final result template uses `spine.call_number` without `spine.index`. Run the focused test and the existing test suite.

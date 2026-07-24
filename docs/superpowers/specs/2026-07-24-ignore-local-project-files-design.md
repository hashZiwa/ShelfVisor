# Local Project Files Ignore Design

## Goal

Keep local documentation and environment-specific project files in the
working tree while removing them from Git tracking.

## Scope

Add root-anchored ignore rules for:

- `/docs/`
- `/requirements.txt`
- `/security_and_structural_risks.txt`

Remove the currently tracked copies from the Git index with
`git rm --cached`. The files must remain present in the local working tree.

## Verification

- `git check-ignore` identifies all three targets.
- `git status` shows `.gitignore` modified and the previously tracked targets
  staged as deletions.
- The ignored files and `docs/` directory still exist locally.
- Existing changes under `app/` are not staged or modified.

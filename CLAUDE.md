# Course Resources

## Goal

Open-source the coursework I have done while self-studying CS and AI: assignment solutions, lab code, and my own notes. The repo should be easy for others to browse and reuse.

## Principles

- **Publish my own work.** Solutions, code, and notes (`Digest/`, `Notes/`) are what the repo is for.
- **Keep third-party material out.** Slides, textbooks, papers, and datasets stay local and gitignored. Link to the official source instead.
- **Keep the repo small.** Do not commit large binaries such as model weights, datasets, videos, archives, or generated web assets. Ignore them before running `git add`.
- **One directory per course**, named `<Institution>-<Code>` (e.g. `Stanford-CS336`). Each course has its own entry page, and the root `README.md` lists every course with its status.
- **Keep courses self-contained.** Each course keeps its own dependencies and build setup. Do not move files across courses.
- **Be careful with history.** Ask before rewriting git history, deleting files, or force-pushing. Keep unrelated changes in separate commits.
- Follow `.editorconfig`. Commit messages use the form `update: <what changed>`.

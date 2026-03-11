# Repo Sync Playbook

This note captures the practical lessons from syncing `Finance-Agents`
to both GitHub and Gitee.

## What happened

The local repo was not in a clean state:

- local `main` had one new commit
- remote `main` already had two newer commits
- the working tree also contained unstaged deletions of tracked large files
- there was an untracked `Qwen3-8B/` directory that did not contain a full 8B model

If we had pushed carelessly, we could have deleted remote files by accident.

## Safe workflow that worked

### 1. Fix network and auth first

Before touching Git history:

- make sure the local proxy is actually wired into `git`, `gh`, and VS Code
- verify GitHub is reachable through the proxy
- complete `gh auth login`

Important detail:

- browser authorization alone is not enough
- `gh auth login` must finish in the terminal and `gh auth status` must show
  `Logged in`

### 2. Never push from a dirty working tree

The original working tree had unstaged tracked deletions:

- model files
- CSV data files

Those files were intentionally kept out of the commit and out of the push.

Rule:

- only push from a clean tree or from an isolated clone/worktree

### 3. Commit only the safe, intended changes

The safe commit included:

- portfolio-style `readme.md`
- `Qwen` text/config references switched to `8B`
- sample report docs
- troubleshooting notes

The risky items were not included:

- unstaged deletions of tracked assets
- incomplete local `Qwen3-8B/` folder

### 4. If remote `main` moved, do not force push

Push to `origin/main` was rejected because remote `main` was ahead.

Correct response:

- fetch remote
- inspect divergence
- determine whether remote changes overlap

In this case, remote changes were two `readme.md` updates, so direct push was
not safe.

### 5. Use an isolated clone for final integration

The reliable solution was:

1. keep the original dirty repo untouched
2. use a clean temporary clone at remote `main`
3. fetch the already-uploaded feature commit
4. `cherry-pick` it onto clean `main`
5. push the integrated result to GitHub `main`
6. push the same integrated result to Gitee `main`

This avoids mixing:

- local experimental files
- accidental deletions
- unfinished assets

### 6. Sync Gitee from the same integrated commit

Gitee should follow the same final commit as GitHub.

Rule:

- do not maintain two independent `main` branches
- integrate once
- push the exact same commit to both remotes

## Practical rules to keep

### Auth and proxy

- prefer `HTTPS` for `gh auth login` when the repo remote is HTTPS
- verify login with `gh auth status`
- if browser auth succeeds but CLI still says not logged in, the terminal flow
  did not finish

### Git safety

- do not push from a dirty repo
- do not use force push unless there is an explicit reason
- if remote has moved, inspect before integrating
- use a temporary clone when the main working tree is risky

### Multi-remote workflow

- keep `origin` as the primary source of truth
- keep `gitee` as a sync target
- after integration, push the same commit to both

## Recommended routine next time

1. make changes in local repo
2. commit only intended files
3. fetch `origin/main`
4. if `main` diverged, integrate in a clean clone
5. push to GitHub `main`
6. push the same commit to Gitee `main`
7. only then clean up local experimental files if needed

## Result of this round

Final integrated commit:

- `6bd21e2`

Synced to:

- GitHub `main`
- Gitee `main`

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 3 ]]; then
  echo "Usage: colab_commit_push.sh <git-name> <git-email> <commit-message> [paths-to-add...]" >&2
  echo "Requires env vars: GITHUB_USER and GITHUB_TOKEN" >&2
  exit 1
fi

GIT_NAME="$1"
GIT_EMAIL="$2"
COMMIT_MESSAGE="$3"
shift 3

if [[ -z "${GITHUB_USER:-}" || -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Set GITHUB_USER and GITHUB_TOKEN before running this script." >&2
  exit 1
fi

git config user.name "$GIT_NAME"
git config user.email "$GIT_EMAIL"

if [[ $# -gt 0 ]]; then
  git add "$@"
else
  git add artifacts
fi

if git diff --cached --quiet; then
  echo "No staged changes to commit."
  exit 0
fi

git commit -m "$COMMIT_MESSAGE"
git remote set-url origin "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/SteveMama/rt-geometry-memory.git"
git push origin main

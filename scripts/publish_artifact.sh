#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PAPER=""
SOURCE_DIR=""
ARTIFACT_NAME=""
COMMIT_MESSAGE=""
DO_PUSH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paper)
      PAPER="$2"
      shift 2
      ;;
    --source)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --name)
      ARTIFACT_NAME="$2"
      shift 2
      ;;
    --commit)
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --push)
      DO_PUSH=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PAPER" || -z "$SOURCE_DIR" ]]; then
  echo "Usage: publish_artifact.sh --paper paper1|paper2|paper3 --source <dir> [--name <artifact-name>] [--commit <msg>] [--push]" >&2
  exit 1
fi

case "$PAPER" in
  paper1|paper2|paper3) ;;
  *)
    echo "Invalid paper bucket: $PAPER" >&2
    exit 1
    ;;
esac

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

if [[ -z "$ARTIFACT_NAME" ]]; then
  ARTIFACT_NAME="$(basename "$SOURCE_DIR")"
fi

DEST_DIR="artifacts/${PAPER}/${ARTIFACT_NAME}"
rm -rf "$DEST_DIR"
mkdir -p "$(dirname "$DEST_DIR")"
rsync -a --delete --exclude '__pycache__' "$SOURCE_DIR"/ "$DEST_DIR"/

echo "Published $SOURCE_DIR -> $DEST_DIR"

if [[ -n "$COMMIT_MESSAGE" ]]; then
  git add "$DEST_DIR"
  if ! git diff --cached --quiet; then
    git commit -m "$COMMIT_MESSAGE"
    if [[ "$DO_PUSH" -eq 1 ]]; then
      git push
    fi
  else
    echo "No tracked changes to commit for $DEST_DIR"
  fi
fi

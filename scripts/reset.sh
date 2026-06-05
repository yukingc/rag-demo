#!/usr/bin/env bash
# scripts/reset.sh — wipe the Chroma database so you can re-ingest from scratch

set -e

CHROMA_DIR="${1:-./chroma_db}"

if [ -d "$CHROMA_DIR" ]; then
  echo "→ Removing $CHROMA_DIR ..."
  rm -rf "$CHROMA_DIR"
  echo "  Done. Run 'python ingest.py' to rebuild."
else
  echo "  Nothing to remove ($CHROMA_DIR does not exist)."
fi

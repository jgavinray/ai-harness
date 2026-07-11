#!/usr/bin/env bash
# Runner calls us with cwd=workdir and $1=this task's directory, after the
# initial commit. Overlay the pending (uncommitted) changes under review.
set -e
cp "$1"/pending_changes/*.py .

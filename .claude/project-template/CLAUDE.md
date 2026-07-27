# Project rules

<!-- Keep under ~40 lines. Local models degrade with long instruction files. -->

## Stack
- Language/toolchain: <fill in, with pinned versions>
- Build: `<command>`
- Test: `<command>`
- Lint/format: `<command>`

## Definition of done
`.claude/verify.sh` exits 0. Nothing else counts.

## Conventions
- <e.g. error handling style, module layout, naming>
- <e.g. never touch generated/ directories>

## Commands you must never run here
- <e.g. terraform apply, kubectl delete, migrations against prod>

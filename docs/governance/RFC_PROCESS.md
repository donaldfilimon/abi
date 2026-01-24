# RFC Process

## Overview

The RFC (Request for Comments) process is how major changes are proposed and discussed in the ABI project.

## When to Submit an RFC

- New features affecting public API
- Breaking changes to existing APIs
- Significant architectural changes
- New sub-systems or modules
- Changes to build system or tooling

## Process

### 1. Pre-RFC Discussion
- Open a GitHub Discussion to gauge interest
- Get initial feedback before formal RFC

### 2. Submit RFC
- Fork the repository
- Copy `RFC_TEMPLATE.md` to `docs/governance/rfcs/XXXX-short-title.md`
- Fill out the template
- Submit a Pull Request

### 3. Review Period
- Minimum 2 weeks for comments
- Author addresses feedback
- Maintainers facilitate discussion

### 4. Decision
- Maintainers vote on approval
- Requires 2/3 majority
- Decision is final

### 5. Implementation
- RFC author or volunteers implement
- Reference RFC in PR description
- Update RFC status when complete

## RFC Lifecycle

```
Draft -> Under Review -> Accepted -> Implemented
                     -> Rejected
```

## Numbering

RFCs are numbered sequentially starting from 0001.

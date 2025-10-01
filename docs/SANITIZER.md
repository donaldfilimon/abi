# Sanitizer Builds

Sanitizer runs are optional but recommended for debugging allocator misuse.
Use the Zig `--sanitizer=address` flag when building locally. Some third-party
libraries may trigger known false positives; document the findings in the
associated pull request and keep the suppression list minimal.

CI does not run sanitizer jobs by default. When enabling them in follow-up
workflows ensure builders have sufficient memory and disable GPU features unless
the platform explicitly supports them.

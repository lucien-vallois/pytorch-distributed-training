# Security Policy

## Supported versions

This project has not published a stable release. Security fixes currently target the latest
commit on the default branch.

## Reporting a vulnerability

Use the repository host's private vulnerability-reporting feature when it is enabled. Include:

- the affected module and revision;
- a minimal reproduction;
- the expected impact;
- any known mitigation.

If no private reporting channel is visible, open a public issue requesting a private contact
channel without including vulnerability details. Do not send secrets, credentials, private data,
or a working exploit in a public issue.

No response-time or release-time guarantee is offered. Maintainers should acknowledge reports,
validate them, prepare a tested fix, and coordinate disclosure according to severity.

## User responsibility

Install dependencies from trusted indexes, pin versions when reproducibility matters, protect
training data and checkpoints, and load checkpoint files only from trusted sources. Distributed
workers communicate over infrastructure controlled by the user; this library does not configure
network authentication or isolation.

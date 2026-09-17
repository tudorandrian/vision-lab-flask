# Security policy

## Supported versions

Only the latest release on the `main` branch receives fixes.

## Reporting a vulnerability

Use GitHub private vulnerability reporting: open the Security tab of this repository and choose
"Report a vulnerability". Do not open a public issue for a security problem. Expect a first
answer within seven days.

## Scope and threat model

vision-lab-flask is a demonstration application. It has no accounts and no authentication, it
binds to the loopback interface by default, and it is not designed to be exposed to the internet
without a reverse proxy that adds TLS, authentication and rate limiting.

Within that scope the application defends against:

| Threat | Defence |
| --- | --- |
| Malicious or oversized uploads, decompression bombs | size limit, pixel limit checked before decoding, format allow-list, full re-encode |
| Path traversal through file names or identifiers | the client never supplies a path; identifiers and file names match strict patterns before any disk access |
| Leaking personal data in uploads | metadata is dropped on re-encode, no gallery or listing, automatic deletion |
| Resource exhaustion | bounded image size, bounded parameters, one inference at a time with a bounded queue, a request body limit enforced by the WSGI server |
| Cross-site scripting | autoescaped templates, no JavaScript, strict Content-Security-Policy |
| Information disclosure on errors | generic error pages; details go to the server log only |
| Supply chain | locked dependencies, weekly automated updates, vulnerability audit and secret scanning in CI, vendored front-end assets |

Out of scope: denial of service by a client that is allowed to upload continuously, and the
statistical accuracy or bias of the pretrained models.

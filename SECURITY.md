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
| Leaking personal data in uploads | metadata is dropped on re-encode, no gallery or listing, deletion at the next accepted upload or the next request for any result after expiry; a job directory left by a crash, with no result, is removed one day after the TTL |
| Resource exhaustion | bounded image size, bounded parameters, one inference at a time, a bounded number of uploads waiting for it, the rest refused at once, a request body limit enforced by the WSGI server |
| Cross-site scripting | autoescaped templates, no JavaScript, strict Content-Security-Policy |
| Cross-site request forgery (another site posting work to a reachable instance) | `POST /jobs` refuses requests whose `Sec-Fetch-Site` is `cross-site`, or whose `Origin` names another host, with HTTP 403; requests without either header (command-line clients) are accepted |
| Information disclosure on errors | generic error pages; details go to the server log only, except a body far over the upload limit, which waitress itself refuses with a plain page naming the configured byte limit, before the request reaches this application |
| Supply chain | locked dependencies, weekly automated updates, vulnerability audit and secret scanning in CI, GitHub Actions pinned to full commit SHAs, vendored front-end assets |

Out of scope: denial of service by a client that is allowed to upload continuously, and the
statistical accuracy or bias of the pretrained models.

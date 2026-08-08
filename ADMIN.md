# Admin Guide

## Accounts & sign-up

By default openkoutsi is **invite-only**: an admin mints an invitation
(`POST /api/admin/invitations`, or the **Invitations** tab in the admin console)
and shares the resulting `…/register?token=…` link with the new user.

Optionally, admins can enable **self-serve email signup** (see below). Invitations
keep working regardless of that toggle.

### Self-serve email signup (optional)

When an email provider is configured (see [DEPLOY.md](DEPLOY.md), *Email*) and the
`allow_self_signup` instance setting is turned on, anyone can register with their
email address:

1. The user submits their email + password on the public sign-up page.
2. openkoutsi creates a **pending** account and emails a
   `…/verify-email?token=…` link.
3. Opening the link verifies the address and activates the account.

Enable it from the **Settings** tab of the admin console, or via the API:

```bash
curl -X PATCH https://api.your-domain/api/admin/settings \
  -H "Authorization: Bearer <admin-access-token>" \
  -H "Content-Type: application/json" \
  -d '{"allow_self_signup": true}'
```

If no email provider is configured, self-serve signup stays unavailable even when
the toggle is on (the sign-up page hides itself), so accounts can never get stuck
un-verifiable.

## Personal access tokens

Users can issue **personal access tokens** to their own tooling — a backup
script, a cron job, a phone, an external MCP client — from **Settings → Personal
access tokens**. A token is scoped, expires, and can be revoked; it grants
strictly *less* than the session its owner already holds. It adds duration, not
authority.

What a token can never do, whatever scopes it was granted:

| Excluded | Why |
|---|---|
| `/api/admin/*` | Admin status must not widen the athlete-data surface. Excluded **even when its owner is an administrator**. |
| `/api/auth/*`, `/api/setup` | A token must never mint or refresh a credential. |
| The token endpoints themselves | A token cannot create, list or revoke a token. Session-authenticated only, and there is no internal minting path. |
| `/api/messages` | The inbox is where expiry warnings and admin-revocation notices land. A credential should not be able to read the message saying it is about to be cut off. |
| `/api/llm/*` and the AI triggers | They spend money. |
| Changing the **LLM configuration** (`llm_base_url`, `llm_api_key`) | Repointing it would make the user's *own* session send their data to a host of the token holder's choosing — closing only the endpoints that spend money would leave that open. |
| Starting a provider **OAuth flow** (`GET /api/integrations/{provider}/connect`) | It mints a signed `state` that the unauthenticated callback trusts to decide whose account the provider tokens are written to. Connecting Strava or Wahoo stays a browser act. |

`GET /api/athlete/export` **is** reachable, but only under its own
`athlete:export` scope — one call that returns the entire record is never folded
into a general read — and every export is written to the audit log by token id.

### Turning the feature off

`allow_personal_access_tokens` is an instance setting, **on by default**. Turning
it off refuses *authentication*, not just issuance: tokens handed out beforehand
stop working immediately, rather than the switch being a comforting untruth.

Enable or disable it from the **Settings** tab of the admin console, or via the
API:

```bash
curl -X PATCH https://api.your-domain/api/admin/settings \
  -H "Authorization: Bearer <admin-access-token>" \
  -H "Content-Type: application/json" \
  -d '{"allow_personal_access_tokens": false}'
```

### Expiry

Every token expires — the picker offers 7 / 30 / 90 / 180 / 365 days, defaults to
90, and the one-year ceiling is enforced server-side, so a hand-rolled request
asking for longer is rejected. There is no "never".

A daily sweep warns the owner 7 days out, 1 day out, and once the token has
expired. The inbox message is unconditional; the matching email is best-effort
(it needs a configured provider and a verified address) and each user can turn it
off from their profile. Each stage fires exactly once per token.

### Inspecting and revoking a user's tokens

When a token is behind a runaway integration, the audit log and rate limits name
it by **token id** — and the instance switch would take down every user while
deleting the account is absurd. So an admin can list and revoke one user's
tokens:

```bash
# List (metadata only — never the token's name)
curl https://api.your-domain/api/admin/users/<user-id>/tokens \
  -H "Authorization: Bearer <admin-access-token>"

# Revoke one
curl -X DELETE https://api.your-domain/api/admin/users/<user-id>/tokens/<token-id> \
  -H "Authorization: Bearer <admin-access-token>"
```

Three deliberate limits on that power:

- **Metadata only, never the name.** Token names are user-written free text and
  can be revealing on their own (`garmin-sync-for-my-cardiologist`). Revocation
  needs the id, not the label.
- **Revoke only — never issue on a user's behalf.** An admin-minted token would
  be indistinguishable from one the user created, which is exactly the failure
  this feature exists to avoid.
- **The user is told.** Every admin revocation lands in their inbox, and in the
  audit log.

This is not a new capability: on a self-hosted instance you already hold
`ENCRYPTION_KEY` and root on the box, and could open `registry.db` yourself. The
endpoint moves the action out of a shell and into the audit log.

### Audit log

Every token-authenticated request is written to the `openkoutsi.audit` logger as
a structured record — token id, user, method, path, required scope and outcome —
rather than to a database shared across users. Outcomes distinguish
`revoked` from `unknown_token`, which is why revoked rows are retained: somebody
using a credential you withdrew is a different event from somebody guessing.

## Password reset

There are two ways to reset a password.

### Self-serve reset by email (when email is configured)

The user clicks **Forgot password?**, enters their email, and — if a verified
account matches — receives a `…/reset-password?token=…` link
(`POST /api/auth/request-password-reset`). The endpoint always returns success and
never reveals whether an account exists, so it can't be used to probe for
addresses.

### Admin-initiated reset (always available)

If email isn't configured, or a user can't receive mail, an admin generates a
reset link and delivers it out-of-band (chat, SMS, etc.):

```bash
curl -X POST https://api.your-domain/api/admin/users/<user-id>/password-reset \
  -H "Authorization: Bearer <admin-access-token>"
```

Response:

```json
{ "reset_url": "https://your-domain/reset-password?token=<token>" }
```

The admin console exposes this as a **Reset password** action on each user in the
**Users** tab, which copies the link to the clipboard.

Send the `reset_url` to the user. The link expires after **1 hour** and is
single-use; generating a new token for a user automatically invalidates any
previous unused token for that user.

Optionally set an `admin_contact` instance setting (Settings tab, or
`PATCH /api/admin/settings`) — it's shown on the "Forgot password?" page (served
via the unauthenticated `GET /api/public/instance-info`) so users know how to
reach you when self-serve reset isn't available.

### User flow

The user visits the link, enters a new password (min 12 chars, at least one
uppercase letter and one digit), and is redirected to the login page.

### Rate limits

- Admin reset-link generation: 10 requests/hour per IP
- Self-serve reset request: 10 requests/hour per IP
- Password reset (token consumption): 10 requests/hour per IP
- Self-serve signup: 10 requests/hour per IP
- Email verification: 20 requests/hour per IP
- Personal access token creation: 20 requests/hour per user

Rate limits are keyed by **principal**: an authenticated request is keyed on the
*user*, everything else on the client address. One script hammering from one
address is not one anonymous visitor. The key is the user rather than the token
because a user may mint tokens freely — per-token buckets would make every limit
multiplicative in a number nothing caps. Token ids still appear in the audit log,
which is where per-token attribution belongs.

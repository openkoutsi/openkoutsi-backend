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

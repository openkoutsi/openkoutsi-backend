# Admin Guide

## Password Reset

openkoutsi does not send email. Password resets are admin-initiated: the admin generates a short-lived reset link and delivers it to the user out-of-band (chat, SMS, etc.).

### Setup

Add `ADMIN_SECRET` to your `.env` (backend):

```env
# .env (backend)
ADMIN_SECRET=<strong-random-string>
```

Generate a value with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Restart the backend after adding the variable. If `ADMIN_SECRET` is not set, the reset endpoint returns 403.

Optionally, show your contact details on the "Forgot password?" page so users know how to reach you. This is an instance setting managed from the admin dashboard (or via `PATCH /api/admin/settings` with an `admin_contact` field) and served to the unauthenticated reset page through `GET /api/public/instance-info`. It can be any string — an email address, a Slack handle, a phone number, etc. If left unset the page just shows the generic "ask your administrator" message.

### Generating a reset link

```bash
curl -X POST https://api.your-domain/api/auth/admin/reset-token \
  -H "X-Admin-Secret: <your-admin-secret>" \
  -H "Content-Type: application/json" \
  -d '{"username": "johndoe"}'
```

Response:

```json
{
  "reset_url": "https://your-domain/reset-password?token=<token>",
  "expires_at": "2026-04-18T15:30:00+00:00"
}
```

Send the `reset_url` to the user. The link expires after **1 hour** and is single-use. Generating a new token for a user automatically invalidates any previous unused token for that user.

### User flow

The user visits the link, enters a new password (min 12 chars, at least one uppercase letter and one digit), and is redirected to the login page.

### Rate limits

- Token generation: 5 requests/hour per IP
- Password reset: 10 requests/hour per IP


# Before Strava Integration Goes Live

These steps require external accounts/services and can't be automated.

---

## 1. Create a Strava API Application

Go to **https://www.strava.com/settings/api** and create an app.

- **App Name:** openkoutsi (or anything you like)
- **Authorization Callback Domain:** `localhost` for dev, your domain for prod

You'll receive a **Client ID** and **Client Secret**. Add them to `.env`:

```
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
```

The existing `openkoutsi_STRAVA_API_TOKEN` in `.env` is from an old personal token
flow — it can be removed once OAuth is working.

---

## 2. Deploy the Strava Bridge

`strava_bridge/` is a separate small service that receives Strava webhooks.
Strava requires a **public HTTPS URL** to send webhooks to.

**Dev (ngrok):**
```bash
# Terminal 1
cd strava_bridge && uv run uvicorn main:app --port 8001

# Terminal 2
ngrok http 8001
# → note the https://xxxx.ngrok.io URL
```

**Prod:** deploy behind a reverse proxy with TLS.

Choose a random secret and add to `.env` in both the main app and the bridge:
```
BRIDGE_URL=https://your-bridge-host
BRIDGE_SECRET=some-long-random-string
```

The bridge reads its own settings from `strava_bridge/.env` (or environment):
```
BRIDGE_SECRET=same_as_above
DATABASE_PATH=bridge.db
```

The bridge does not need `STRAVA_CLIENT_SECRET`: `POST /webhook` is
unauthenticated, because Strava documents no webhook signing. See "Webhook
authentication" in `DEPLOY.md`.

---

## 3. Register the Strava Webhook Subscription

After the bridge is publicly reachable, register with Strava once:

```bash
curl -X POST https://www.strava.com/api/v3/push_subscriptions \
  -F client_id=YOUR_CLIENT_ID \
  -F client_secret=YOUR_CLIENT_SECRET \
  -F callback_url=https://your-bridge-host/webhook \
  -F verify_token=YOUR_BRIDGE_SECRET
```

Strava will call `GET /webhook` on the bridge with a challenge. If the bridge
responds correctly you'll get a `{"id": N}` back — that's the subscription ID.
Keep it for future reference (to delete/update the subscription).

---

## Summary of `.env` keys needed for Phase 2

| Key | Where set | Description |
|-----|-----------|-------------|
| `STRAVA_CLIENT_ID` | main app `.env` | From Strava app settings |
| `STRAVA_CLIENT_SECRET` | main app `.env` | From Strava app settings. Also needed in the bridge `.env` only if `STRAVA_VERIFY_WEBHOOK_SIGNATURE=true` |
| `BRIDGE_URL` | main app `.env` | Public URL of the bridge service |
| `BRIDGE_SECRET` | main app `.env` + bridge `.env` | Shared secret you choose |

---

# Before Hammerhead (Karoo) Integration Goes Live

Issue #69. Also optional — an instance that leaves these unset simply does not
offer Hammerhead.

1. **Register an app** in the Hammerhead developer dashboard. Redirect URI:
   `https://<API_URL>/api/integrations/hammerhead/callback`. Scope:
   `activity:read` (read-only; nothing is pushed to the Karoo).
2. **Deploy `hammerhead_bridge/`** behind public HTTPS (see DEPLOY.md §5b) and set
   its webhook URL (`https://<bridge-host>/webhook`) and webhook secret in the
   dashboard.
3. **Confirm empirically** what the published spec leaves open: rate limits (each
   new activity costs a detail call plus a FIT download), the units of
   `ActivitySummary.duration`/`distance` (not used — the FIT is authoritative),
   the encoding of `X-Hmac-Signature` (the bridge accepts hex or base64), whether
   the token response lists granted scopes, and whether the API licence requires
   visual attribution.

| Key | Where set | Description |
|-----|-----------|-------------|
| `HAMMERHEAD_CLIENT_ID` | main app `.env` | From the Hammerhead developer dashboard |
| `HAMMERHEAD_CLIENT_SECRET` | main app `.env` | From the Hammerhead developer dashboard |
| `HAMMERHEAD_BRIDGE_URL` | main app `.env` | URL of the Hammerhead bridge, as reachable from the main app |
| `HAMMERHEAD_BRIDGE_SECRET` | main app `.env` + bridge `.env` | Shared secret you choose, ≥ 32 characters |
| `HAMMERHEAD_WEBHOOK_SECRET` | bridge `.env` | The webhook signing secret from the dashboard |

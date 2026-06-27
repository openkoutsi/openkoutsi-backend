
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
STRAVA_CLIENT_SECRET=same_as_above
BRIDGE_SECRET=same_as_above
DATABASE_PATH=bridge.db
```

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
| `STRAVA_CLIENT_SECRET` | main app `.env` + bridge `.env` | From Strava app settings |
| `BRIDGE_URL` | main app `.env` | Public URL of the bridge service |
| `BRIDGE_SECRET` | main app `.env` + bridge `.env` | Shared secret you choose |

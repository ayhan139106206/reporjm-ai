# Monster Chat API — Cloudflare Worker

Cloudflare Worker API gateway for the Monster Chat project. No pretrained model and no external
AI inference API are used.

## Expiration date
The Worker checks UTC time and returns HTTP 410 after `8002-11-04T00:00:00Z`.
This is an application-level shutdown gate. It does not cancel the Cloudflare deployment,
guarantee free hosting until that date, or override Cloudflare plan limits and policies.

## Endpoints
- `GET /v1/health`
- `GET /v1/model`
- `POST /v1/chat`

`/v1/chat` returns `503 model_engine_not_ready` until the custom Persian model is implemented,
trained from random initialization, and connected. It does not fabricate an AI response.

## GitHub Actions deployment
Add repository Actions secrets:
- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

Push to `main`; GitHub Actions deploys the Worker and then stops. Cloudflare serves the Worker
independently from the GitHub runner.

## Optional authentication
Set Worker secret `API_KEY` in Cloudflare and send:
`Authorization: Bearer YOUR_API_KEY`

## Free plan caveat
Workers Free has usage limits and runtime restrictions. It is suitable for this lightweight
JavaScript API gateway, not for PyTorch model training/inference. The custom model needs a
separate compute environment or a compatible implementation within Workers limits.

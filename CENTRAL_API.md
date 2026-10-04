# Central observation API

This repository now contains a persistent HTTP API backed by SQLite. The
database is the authority for duplicate detection; each accepted batch runs in
one `BEGIN IMMEDIATE` transaction and the table's composite primary key is
`(normalized JPC, normalized observation text)`. There is no generated row ID.

## Railway production preparation

`.railway/railway.ts` declares the FastAPI service, one persistent volume, the
deployment command, and its health check using Railway's current Infrastructure
as Code format. This repository is prepared but has not been deployed. Connect
the central service to this repository before deploying it.

Before the first deployment:

1. Review the IaC volume region and 1 GB size, then run `railway config plan`
   and review it before applying. The declared persistent volume is mounted at
   `/data`. Railway mounts volumes at runtime; the API refuses to start in a
   Railway environment when the volume is absent or the database path is
   outside its mount.
2. Set `PDI_CENTRAL_DB_PATH=/data/observations.sqlite3`.
3. Set `PDI_CENTRAL_API_TOKEN` to a secret generated in Railway with at least
   32 characters. Keep the same secret in the Streamlit client service.
4. Generate the central API service's public HTTPS domain and set
   `PDI_CENTRAL_API_URL` to that base URL in the Streamlit client service.
5. Keep the central API at one replica/instance. SQLite serializes writes on
   its host, but this store is not a shared multi-host database.

`.env.railway-api.example` and `.env.railway-client.example` list the variables
for their respective services. The IaC uses `preserve()` for the API token so
the real secret must be entered in Railway's service variables before deploy.
Configure real values in Railway's service variable settings; never commit
production secrets. `/health` checks database
availability and returns only `{"status":
"ok"}` or a generic unavailable response. The Railway health check is not an
authenticated data route.

## Configure and run locally

Run one service instance with persistent local disk. Set:

```powershell
$env:PDI_CENTRAL_DB_PATH = 'D:\PDI-Central\observations.sqlite3'
$env:PDI_CENTRAL_API_TOKEN = '<long random secret from your secret manager>'
$env:PDI_CENTRAL_API_HOST = '127.0.0.1'
$env:PDI_CENTRAL_API_PORT = '8765'
python central_api.py
```

For local development, the stdlib HTTP compatibility server remains available
through `python central_api.py`. Railway runs the FastAPI ASGI app using the
command in `railway.json`. Expose external clients only over HTTPS. Back up the
persistent database. SQLite locking is appropriate for one central service
host; a multi-host deployment should migrate this same composite-key/transaction
contract to a shared transactional database such as PostgreSQL before running
replicas.

## Configure the Streamlit client

Set `PDI_CENTRAL_API_URL` to the HTTPS base URL and `PDI_CENTRAL_API_TOKEN` to
the same secret in the Streamlit service environment. Without both values, the
adapter returns `ERROR` and the app labels any report as a local preview.

The write endpoint is `POST /v1/observations/batch`. Its JSON request contains
exactly `jpc`, `inspector`, and a non-empty `observations` list. The service
returns `SUCCESS`, `DUPLICATE`, `REJECTED`, or `ERROR`. `GET /v1/observations?jpc=…`
provides authenticated read-back for operational verification. Both routes
require `Authorization: Bearer …`.

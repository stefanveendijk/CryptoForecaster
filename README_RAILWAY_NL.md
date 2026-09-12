# Crypto Forecaster – permanente cloudbackend via Railway

Dit pakket draait Ultimate v4 + FastAPI als één Railway-service.

## Wat gebeurt automatisch?

- API is permanent bereikbaar via HTTPS.
- Eerste modelrun start na deployment automatisch op de achtergrond.
- Daarna start dagelijks na 04:15 UTC een nieuwe modelrun.
- `/data` bewaart modelcache en output op een Railway volume.
- De iPhone-app blijft exact dezelfde API gebruiken.

## Kosten

Railway Free heeft weinig RAM. Dit model kan meer geheugen nodig hebben.
Voor betrouwbaar gebruik is Railway Hobby waarschijnlijk passender.

## Installeren

Dubbelklik:

`12_DEPLOY_NAAR_RAILWAY.bat`

Het script installeert Railway CLI en begeleidt de deployment.

## Na deployment

Test:

`https://<jouw-domein>.up.railway.app/api/v1/health`

Vul daarna in de iPhone-app bij Instellingen alleen in:

`https://<jouw-domein>.up.railway.app`

## Eerste modelrun

`modelOutputExists:false` direct na deployment is normaal.

Bekijk:
- `/api/v1/run-status`
- `/api/v1/log-tail`

Zodra `modelOutputExists:true` en `lastSuccess` een datum bevat, is de cloudmodeloutput klaar.

## Persistent volume

Controleer in Railway Dashboard dat het volume op `/data` is gekoppeld.
Zonder volume gaan cache en output bij redeploy verloren.

## Dagelijkse instellingen

Railway Variables:
- `MODEL_HOUR_UTC=4`
- `MODEL_MINUTE_UTC=15`
- `MODEL_NEWS=true`
- `MODEL_DEEP=false`
- `AUTO_RUN_MODEL=true`

Laat `MODEL_DEEP=false` totdat de balanced cloudversie stabiel draait.

# Hosting the public demo on Azure App Service

Live: https://godseye-kolkata-fpech9f9drfwfrf7.uaenorth-01.azurewebsites.net (UAE North; Central India is not open to this student subscription).

The read-only demo runs as a container on one Basic (B1) Linux web app: 1.75 GB
of memory, always on, at about US$13 a month. An *Azure for Students*
subscription carries US$100 of credit with no card and a spending limit, so it
covers roughly seven months and can never bill anyone; when the credit runs
out the app simply stops.

## The image

`.github/workflows/demo-image.yml` builds `deploy/Dockerfile` on every push to
`main` and publishes it as `ghcr.io/a-martyr/godseye:latest`. Check the run
under the repository's **Actions** tab (about 10 minutes the first time).

The package takes the public repository's visibility, so App Service can pull
it without a password. If it ever shows as private, make it public: GitHub
profile > **Packages** > `godseye` > **Package settings** > *Danger Zone* >
**Change visibility** > Public.

## The web app (Azure portal)

1. **Create a resource** > **Web App**.
2. *Basics*: subscription **Azure for Students**; resource group: new,
   `godseye`; name: `godseye` plus anything that makes it unique (it becomes
   the address); **Publish: Container**; **Operating system: Linux**; region:
   the nearest one the subscription allows (student subscriptions allow only a
   few; if one is refused, try another); pricing plan: new, **Basic B1**.
3. *Container*: image source **Other container registries**; access **Public**;
   registry server `https://ghcr.io`; image and tag `a-martyr/godseye:latest`;
   startup command empty. The newer *sidecar* form asks for the port on this
   tab: set **Port** to `7860`.
4. **Review + create** > **Create**, then **Go to resource**. If Azure refuses
   the region, list the ones the subscription allows in Cloud Shell with
   `az policy assignment list --disable-scope-strict-match --query
   "[].parameters.listOfAllowedLocations.value"`.
5. **Settings > Environment variables** > add
   `WEBSITES_CONTAINER_START_TIME_LIMIT` = `600` (and `WEBSITES_PORT` = `7860`
   if the form did not ask for the port); **Apply**.
6. **Settings > Configuration > General settings**: **Always on: On**,
   **Web sockets: On** (the live feed is a websocket); **Save**.
7. **Overview** > **Restart**. The first start pulls the image and seeds the city;
   allow two or three minutes, then open the **Default domain**.

The app is read-only on its public address automatically (`core/reach.py`
treats any hostname as public) and `GODSEYE_PUBLIC_DEMO=1` is baked into the
image.

## After a push

GitHub Actions publishes a new image; **Restart** the web app to pull it.

## When it is no longer needed

Delete the `godseye` resource group (**Resource groups** > `godseye` > **Delete
resource group**). That stops all spending at once.

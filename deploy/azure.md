# Hosting the public demo on Azure App Service

The read-only demo runs as a container on one Basic (B1) Linux web app: 1.75 GB
of memory, always on, at about US$13 a month. An *Azure for Students*
subscription carries US$100 of credit with no card and a spending limit, so it
covers roughly seven months and can never bill anyone; when the credit runs
out the app simply stops.

## The image

`.github/workflows/demo-image.yml` builds `deploy/Dockerfile` on every push to
`main` and publishes it as `ghcr.io/a-martyr/godseye:latest`. Check the run
under the repository's **Actions** tab (about 10 minutes the first time).

The first publish creates the package as **private**. Make it public once:
GitHub profile > **Packages** > `godseye` > **Package settings** > *Danger
Zone* > **Change visibility** > Public. App Service can then pull it without a
password.

## The web app (Azure portal)

1. **Create a resource** > **Web App**.
2. *Basics*: subscription **Azure for Students**; resource group: new,
   `godseye`; name: `godseye` plus anything that makes it unique (it becomes
   the address); **Publish: Container**; **Operating system: Linux**; region:
   the nearest one the subscription allows (student subscriptions allow only a
   few; if one is refused, try another); pricing plan: new, **Basic B1**.
3. *Container*: image source **Other container registries**; access **Public**;
   registry server `https://ghcr.io`; image and tag `a-martyr/godseye:latest`;
   startup command empty.
4. **Review + create** > **Create**, then **Go to resource**.
5. **Settings > Environment variables** > add `WEBSITES_PORT` = `7860` and
   `WEBSITES_CONTAINER_START_TIME_LIMIT` = `600`; **Apply**.
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

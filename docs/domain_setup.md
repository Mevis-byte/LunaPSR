# LunaPSR — Deployment & Domain Setup

This document explains how LunaPSR reaches the public internet, and exactly
which manual steps remain for the desired custom domain
**`lunapsr.is-a.dev`**.

**Current status: NOT REGISTERED — deployment pending.**
No live URL is claimed anywhere in this repository until the deployment has
actually been created and verified.

---

## 1. Target architecture

```
LunaPSR (this repository, github.com/Mevis-byte/LunaPSR)
        │  deploy
        ▼
Streamlit Community Cloud            ← free hosting for the app itself
        │  subdomain chosen during deployment: "lunapsr"
        ▼
https://lunapsr.streamlit.app        ← the actual hosting URL
        │
        │  is-a.dev DNS registration (manual, GitHub-PR based)
        ▼
lunapsr.is-a.dev                     ← desired public name
```

## 2. Step 1 — Deploy to Streamlit Community Cloud

The application entrypoint is **`app.py`** (verified: it contains the Streamlit
page config, imports all page modules and is the documented launch target,
`streamlit run app.py`). Runtime dependencies are declared in
`requirements.txt`.

Manual steps:

1. Sign in at <https://share.streamlit.io> with GitHub (connect the
   `Mevis-byte` account).
2. **Create app → Deploy a public app from GitHub.**
   * Repository: `Mevis-byte/LunaPSR`
   * Branch: `main`
   * Main file path: `app.py`
3. Set the **custom subdomain** to `lunapsr`, giving:
   `https://lunapsr.streamlit.app`

> ⚠ **Important limitation (verified against current Streamlit documentation
> and community forums, 2026):** Streamlit Community Cloud supports only
> `*.streamlit.app` subdomains. It does **not** support attaching an arbitrary
> custom domain such as `lunapsr.is-a.dev` directly to the app — CNAME/A
> records pointed at Streamlit Cloud will not serve the app (TLS/platform
> limitation). Do not invent a CNAME target for Streamlit; there is none.
>
> The practical pattern used by the community is a lightweight static landing
> page on the custom domain that links to — or frames — the Streamlit app via
> its `?embed=true` parameter. That is what step 3 below prepares. Note the
> trade-off honestly: with an iframe embed the browser address bar does not
> track in-app navigation.

## 3. Step 2 — Register `lunapsr.is-a.dev` at is-a.dev

is-a.dev registers subdomains through a GitHub pull request to the
[`is-a-dev/register`](https://github.com/is-a-dev/register) repository. The
official procedure (from the is-a.dev README / <https://docs.is-a.dev>, which
is the authoritative source):

1. **Fork** <https://github.com/is-a-dev/register>.
2. Add a JSON file at `domains/lunapsr.json` following their documented
   record schema.
3. Open a **pull request** back to `is-a-dev/register`.
4. Watch the PR — maintainers may request changes; unaddressed change requests
   lead to rejection.
5. After merge, DNS records publish within minutes.

> Per project policy, **no PR is opened against is-a-dev automatically** — this
> requires explicit authorization. A ready-to-submit JSON template is provided
> below but lives only in these docs, not as part of the LunaPSR deployment.

### 3.1 Ready-to-submit template (documentation only)

Two realistic options, depending on where the public name should land:

**Option A — landing page on GitHub Pages (recommended).** Host a small static
landing page in a separate repo (`Mevis-byte.github.io/lunapsr/` or a repo with
Pages enabled), then point the subdomain at GitHub Pages. The CNAME target for
GitHub Pages is documented by GitHub itself as `<username>.github.io`:

```json
{
  "owner": {
    "username": "Mevis-byte",
    "repo": "lunapsr-landing"
  },
  "record": {
    "CNAME": "Mevis-byte.github.io"
  },
  "proxied": false
}
```

(The landing page repo must contain a `CNAME` file with `lunapsr.is-a.dev`,
per GitHub Pages custom-domain rules, and can link to or iframe-embed
`https://lunapsr.streamlit.app/?embed=true`.)

**Option B — direct redirect service.** If you prefer the domain itself to be
the app address, use a redirect provider that offers you an exact CNAME/A
target and take the value **from that provider's dashboard** — do not guess it.

```json
{
  "owner": { "username": "Mevis-byte" },
  "record": {
    "CNAME": "<OBTAIN FROM HOSTING PROVIDER'S CUSTOM-DOMAIN PANEL>"
  },
  "proxied": false
}
```

Field names follow the is-a.dev docs at registration time — re-check
<https://docs.is-a.dev> before submitting, as their schema may evolve.

## 4. Step 3 — Verify

Checklist before any "live" claim is made anywhere in this repository:

- [ ] `https://lunapsr.streamlit.app` loads the LunaPSR header (dark theme).
- [ ] OHRC product discovery works on the deployed instance *only if* a product
      is bundled — the 1.2-billion-pixel dataset is **not** in the repository,
      so cloud deployments run on the bundled sample image by design.
- [ ] PR to `is-a-dev/register` merged; DNS resolves
      (`nslookup lunapsr.is-a.dev`).
- [ ] Landing page (if Option A) serves over HTTPS with the custom hostname.
- [ ] Update README §Deployment status from *pending* to *live*, with URLs.

## 5. Summary of remaining manual actions

| # | Action | Owner |
|---|---|---|
| 1 | Create Streamlit Community Cloud account + deploy `main@app.py`, subdomain `lunapsr` | manual |
| 2 | Fork `is-a-dev/register`, add `domains/lunapsr.json`, open PR | manual (explicitly authorized action required) |
| 3 | Address PR review comments if any | manual |
| 4 | Post-merge DNS verification + README status update | manual |

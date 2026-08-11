# LogisChain Lab

A dual-domain AI system design — fusing supply chain intelligence with financial risk models for trade finance, working capital, SCF, and credit risk — plus **LogisChain Lab**, a playable gamified simulator built on the same model.

## What's in this repo

| Path | What it is |
|---|---|
| `docs/DESIGN.md` | Full system design: the LogisChain Intelligence architecture (logistics network modeling, shipment risk prediction, disruption forecasting, carrier reliability analytics, fused into trade finance / working capital / SCF / credit risk models) and the LogisChain Lab platform spec |
| `docs/index.html` | Self-contained, dependency-free interactive prototype of LogisChain Lab. Run a $10M trade finance + SCF book through 12 simulated quarters of shipping-lane disruptions |

## Run it locally

No build step — it's a single static HTML file with no dependencies.

```bash
# option 1: just open it
open docs/index.html          # macOS
xdg-open docs/index.html      # Linux
start docs/index.html         # Windows

# option 2: serve it
cd docs && python3 -m http.server 8000
# then visit http://localhost:8000
```

## Live demo via GitHub Pages

Once this repo is pushed to GitHub:

1. **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: **main**, folder: **/docs**
4. Save — your simulator will be live at `https://<your-username>.github.io/<repo-name>/` within a minute or two.

## Push to GitHub

The repo is already initialized and staged locally. From inside the `logischain-lab/` folder:

```bash
git commit -m "Initial commit: LogisChain Intelligence design + LogisChain Lab simulator"
```

Then, with the [GitHub CLI](https://cli.github.com/) (creates the remote repo and pushes in one step):

```bash
gh repo create logischain-lab --public --source=. --remote=origin --push
```

Or without the CLI — create an empty repo named `logischain-lab` at github.com/new first (don't initialize it with a README), then:

```bash
git remote add origin https://github.com/<your-username>/logischain-lab.git
git branch -M main
git push -u origin main
```

## Project structure

```
logischain-lab/
├── README.md
├── LICENSE
├── .gitignore
└── docs/
    ├── index.html   ← interactive simulator (GitHub Pages entry point)
    └── DESIGN.md    ← dual-domain AI system + platform design
```

## Tech

Vanilla HTML/CSS/JS. No dependencies, no build tooling, no backend — all simulator state lives in memory in the browser tab.

## License

MIT — see `LICENSE`. Swap in your organization's standard license if this moves from prototype to an internal or commercial project.

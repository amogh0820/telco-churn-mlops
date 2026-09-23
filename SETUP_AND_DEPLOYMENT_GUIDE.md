# Setup & deployment guide (Windows, beginner-friendly)

This walks you from the unzipped project folder on your PC to a public URL
on Render that you can put on your resume. Every command is exact and goes
in **PowerShell** unless a step says otherwise. Where this guide gives a
plain command instead of a `make` shortcut, that's deliberate — Windows
doesn't have `make` installed by default, and this guide is written so you
never need to install it.

Time: roughly 60-90 minutes the first time, most of it waiting for installs
and the training run, not typing.

---

## 0. What you'll end up with

- The project running on your own PC: a trained model, a working API, a
  working web page.
- The project on GitHub, publicly visible.
- The project live on Render at a URL like
  `https://telco-churn-api-xxxx.onrender.com` — this is your resume link.

---

## 1. Install the software

Install these in order. Each link goes to the official source.

| Software | Get it from | Notes |
|---|---|---|
| **Python 3.11** | <https://www.python.org/downloads/> | Scroll to a 3.11.x release, not 3.12/3.13 — this project's dependencies are tested against 3.11. **On the installer's first screen, tick "Add python.exe to PATH" before clicking Install.** This is the single most common thing beginners miss, and it makes `python` not work in the terminal afterward. |
| **Git for Windows** | <https://git-scm.com/download/win> | Default options are fine. This also installs **Git Bash**, which you'll use once, later, and **Git Credential Manager**, which handles GitHub login for you automatically. |
| **VS Code** | <https://code.visualstudio.com/> | Default options are fine. |
| **Node.js LTS** | <https://nodejs.org/> | Optional — only needed to run the frontend's test file. Skip this if you just want to get the project running. |
| **Docker Desktop** | <https://www.docker.com/products/docker-desktop/> | **Optional.** You do *not* need Docker to deploy to Render — Render builds the container for you in the cloud. Only install this if you want to test the exact production container on your own PC first. It requires a free Docker account (no payment) and will ask to enable WSL2; accept that and restart if it asks. |

You'll also need two free accounts, no payment method for either:
- **GitHub** — <https://github.com/join>
- **Render** — <https://dashboard.render.com/register> (sign up with your GitHub account to save a step later)

**Verify the installs.** Open PowerShell (press Start, type `PowerShell`,
open "Windows PowerShell") and run:

```powershell
python --version
git --version
```

You should see `Python 3.11.x` and a git version number. If `python` says
"not recognized," Python wasn't added to PATH — reinstall it and tick that
box, or search "edit environment variables for your account" in Windows
and add Python's install folder and its `Scripts` subfolder to PATH by hand.

---

## 2. Get the project onto your PC and open it in VS Code

1. Unzip `telco-churn-mlops.zip` somewhere you'll remember, e.g.
   `C:\Users\<you>\Projects\`. You should get a folder named `churn-mlops`.
2. Open **VS Code**.
3. **File → Open Folder…**, select the `churn-mlops` folder, click
   **Select Folder**.
4. Open the built-in terminal: **Terminal → New Terminal** (or
   `` Ctrl+` ``). A panel opens at the bottom. It should already be sitting
   *inside* the `churn-mlops` folder — check the prompt shows
   `...\churn-mlops>`. Every command in this guide runs in this terminal.

If the terminal opens as `cmd.exe` instead of PowerShell, click the small
dropdown arrow next to the `+` in the terminal panel and choose
**PowerShell**.

---

## 3. Create and activate the Python environment

Still in that terminal:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. That means every `python` and
`pip` command from here on uses this project's isolated environment, not
your system Python.

**If you get a red error** like *"running scripts is disabled on this
system"* — PowerShell blocks script execution by default. Run this once
(it only affects your own user, not the whole system):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Type `Y` if it asks, then run the `Activate.ps1` line again.

**Every time you come back to this project in a new terminal**, re-run
`.venv\Scripts\Activate.ps1` first before any other command in this guide.

---

## 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

This installs everything: scikit-learn, pandas, FastAPI, MLflow, pytest,
ruff, Jupyter. It's a few hundred MB and takes a few minutes.

If a package fails to build with a wall of red compiler errors, it's almost
always one specific package that needs a newer pip — re-run the `--upgrade
pip` line above, then the install line again.

---

## 5. Get the real dataset and train the model

This is the step that matters most: the project currently ships with
**no trained model** on purpose (see the main `README.md` for why). This is
the first thing that must succeed.

```powershell
python -m src.data.download
python -m src.data.preprocess
```

The first command downloads the real IBM Telco churn dataset (7,043
customers) from GitHub and checks its shape. The second cleans it and
writes a stratified train/test split to `data\processed\`.

**If this fails with a network or SSL error**, you're likely behind a
firewall or VPN that blocks the download — try a different network, or
open the URL printed in the error directly in a browser to check the file
is reachable.

Now train:

```powershell
python -m src.models.train
```

This runs a real hyperparameter search across three model families with
5-fold cross-validation. **It takes a few minutes** — you'll see log lines
as it goes. When it finishes, you'll have two new files:
`artifacts\model.joblib` (the trained pipeline) and
`artifacts\model_metadata.json` (its metrics and the threshold it chose).

This also writes to a local `mlruns\` folder — that's MLflow's tracking
data, covered in section 9.

---

## 6. Generate the results for the README

```powershell
python -m scripts.update_readme_results
```

This reads `reports\test_report.json` (written by the training step above)
and rewrites the **Results** section of `README.md` with the real numbers
from your run — model comparison, precision/recall, the confusion matrix,
campaign value. Open `README.md` in VS Code afterward and look at that
section; it should no longer say "Not generated yet."

---

## 7. Run the tests

```powershell
pytest tests/ -v --cov=src --cov-report=term-missing
```

You should see a wall of green `PASSED` lines and a coverage summary at the
end, no `FAILED`. If you installed Node.js in step 1, also run:

```powershell
node demo\app.test.js
```

This tests the frontend's logic — the same request-building and
response-formatting code the web page uses, checked against the real API's
request/response shapes.

---

## 8. Run the backend and try the frontend

```powershell
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Leave this running. Open a browser to:

- **http://localhost:8000/** — the scoring page itself. Fill in the form
  (it starts with sensible defaults) and click **Score customer**. You
  should see a real response on the right, including the actual JSON.
- **http://localhost:8000/docs** — the interactive API documentation,
  auto-generated by FastAPI. You can try every endpoint here too.
- **http://localhost:8000/health** — should show `"status":"ok"`.

If Windows Firewall pops up asking to allow Python to accept connections,
click **Allow** — that's normal for running a local server.

Press `Ctrl+C` in the terminal to stop the server when you're done.

---

## 9. Look at MLflow (optional, but worth seeing)

In a **new** terminal (Terminal → New Terminal — remember to activate the
venv again: `.venv\Scripts\Activate.ps1`):

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Open **http://localhost:5000** — you'll see the training run you did in
step 5, with every candidate model's parameters and cross-validated score,
and which one got registered. `Ctrl+C` to stop it.

---

## 10. Docker (optional — skip this if you just want to deploy)

Render builds your Docker image for you in the cloud; you don't need
Docker locally to deploy. Do this section only if you want to test the
exact production container yourself first. Requires Docker Desktop from
step 1, running (check its whale icon in the system tray is not animating).

```powershell
docker build -t telco-churn-api:latest .
docker run --rm -p 8000:8000 --name churn-api telco-churn-api:latest
```

Open http://localhost:8000/ the same way as step 8. `Ctrl+C` to stop it,
then `docker rm -f churn-api` if it's still listed under `docker ps -a`.

If this build fails immediately with something like *"COPY
artifacts/model.joblib: not found"* — you skipped step 5. Train the model
first; the Dockerfile deliberately refuses to build without it.

---

## 11. Create the GitHub repository and push

1. Go to <https://github.com/new>.
2. **Repository name:** something like `telco-churn-mlops`.
3. Leave it **Public** (a private repo won't show on your resume the same
   way, and there's nothing sensitive in this project — see the README's
   "GitHub readiness" section for exactly what was checked).
4. **Do not** tick "Add a README" or ".gitignore" — this project already
   has both, and GitHub will refuse to let you push if the new repo isn't
   empty.
5. Click **Create repository**. Leave the page open — the next screen
   shows the repository's URL, which you need below.

Back in your VS Code terminal (still inside `churn-mlops`, venv doesn't
matter for git commands):

```powershell
git init
git add .
git commit -m "Initial commit: telco churn MLOps project"
git branch -M main
git remote add origin https://github.com/<your-username>/telco-churn-mlops.git
git push -u origin main
```

Replace `<your-username>` and the repo name with your actual values from
the GitHub page.

**The first `git push` will open a browser window** asking you to sign in
to GitHub — this is Git Credential Manager, installed automatically with
Git for Windows in step 1. Sign in there once; it remembers you after.

**Double-check the model made it in.** After pushing, refresh the GitHub
repository page in your browser and confirm you can see
`artifacts/model.joblib` listed as a real file (not just `.gitkeep`). If
it's missing, `.gitignore`'s exception for that file didn't take — run
`git status` and check `artifacts/model.joblib` isn't listed under
"Untracked" as ignored; if it is, something's wrong with `.gitignore` and
you should stop and get help before deploying, since Render's build will
fail the same way.

---

## 12. Deploy on Render

1. Go to <https://dashboard.render.com>, sign in.
2. Click **New +** → **Web Service**.
3. Connect your GitHub account if you haven't, then select your
   `telco-churn-mlops` repository.
4. Render scans the repo. It should offer **Docker** as the detected
   runtime (because of the `Dockerfile` at the repo root). If it suggests
   anything else, change the **Runtime** dropdown to **Docker** by hand.
5. Fill in:
   - **Name:** anything you like — it becomes part of your URL
     (`https://<name>.onrender.com`).
   - **Region:** whichever is closest to you; it doesn't affect anything
     else in this guide.
   - **Instance Type:** **Free**.
6. Expand **Advanced** (or scroll to it) and set:
   - **Health Check Path:** `/health`
   - **Environment Variables** — click **Add Environment Variable** twice:
     - `PORT` = `8000`
     - `WORKERS` = `1`
7. Click **Create Web Service**.

Render starts building immediately — you'll see live logs. The first build
takes a few minutes (installing scikit-learn, pandas, etc. from scratch).
Watch for the log line that says the service is **Live**.

**If the build fails**, scroll the logs up to the actual error — the two
most likely causes are covered in the Troubleshooting section below
(missing model file, or a dependency version issue).

---

## 13. Get your Live Demo URL

Once the dashboard shows **Live** (a green dot), your URL is shown at the
top of the service page, e.g. `https://telco-churn-api-ab12.onrender.com`.

Open it. You should see the same scoring page you tested locally in step
8 — except now the **API base URL** field fills in with that exact Render
URL automatically (same-origin default, no configuration needed). Try a
prediction.

Now put that URL where it belongs:

1. Open `README.md` in VS Code.
2. Find the line near the top: `**Live demo:** _add your Render URL
   here...`. Replace the placeholder with your real URL.
3. Also fill in the **Repository** line with your GitHub URL.
4. Save, then commit and push:

```powershell
git add README.md
git commit -m "Add live demo and repository links"
git push
```

Render redeploys automatically on every push to `main` — but this
particular change doesn't affect the running app, only the README, so
there's nothing to re-check here beyond confirming the push succeeded.

---

## 14. Verify everything is actually working

Check each of these against your **Render URL** (not localhost):

- [ ] `https://<your-app>.onrender.com/` loads the scoring page
- [ ] Submitting the form returns a real prediction (not an error)
- [ ] `https://<your-app>.onrender.com/health` returns `"status":"ok"`
- [ ] `https://<your-app>.onrender.com/docs` loads the API documentation
- [ ] Your GitHub repository page shows all the project files, including
      `artifacts/model.joblib`
- [ ] The README's **Live demo** and **Repository** links are filled in
      (view the README on GitHub, not just locally, to confirm the links
      themselves work)

**Remember: if the service has been idle for 15+ minutes, the first
request takes 30-60 seconds** while Render wakes it back up — this is
normal for the free tier, not a bug. Don't panic if the first click seems
to hang.

---

## 15. Updating and redeploying later

Whenever you change code:

```powershell
git add .
git commit -m "describe what changed"
git push
```

Render watches your GitHub repo and rebuilds automatically on every push
to `main` — nothing else to do. Watch the **Events** tab on the Render
dashboard to see the new deploy happen.

**If you retrained the model** (`python -m src.models.train` again), the
two files in `artifacts/` changed on disk — make sure `git add .` actually
picks them up (`git status` should show them as modified) before you
commit, or Render will keep serving the old model.

---

## 16. Troubleshooting

**`python` or `pip` "is not recognized"**
Python wasn't added to PATH during install. Reinstall from python.org and
tick "Add python.exe to PATH," or add it to PATH manually via Windows'
"Edit environment variables for your account" settings panel.

**PowerShell says running scripts is disabled**
Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
once, then retry. See step 3.

**`pip install` fails with compiler errors**
Run `python -m pip install --upgrade pip` first, then retry the install.
If one specific package keeps failing, check that you're using Python
3.11, not 3.12/3.13 — some pinned versions may not have prebuilt wheels
for the newest Python yet.

**`git push` asks for a password and rejects it**
GitHub no longer accepts your account password for git operations. A
browser sign-in window should appear automatically (Git Credential
Manager) — if it doesn't, create a Personal Access Token at
<https://github.com/settings/tokens> (classic token, `repo` scope) and
paste that in as the password when prompted.

**Render build fails: "COPY artifacts/model.joblib: file not found"**
The model wasn't committed to GitHub. Run `python -m src.models.train`
locally, then `git add artifacts/model.joblib artifacts/model_metadata.json`,
commit, and push again.

**Render build succeeds but the site shows "Model not loaded" or `/predict`
returns a 503**
Check the **Logs** tab on the Render dashboard for the actual startup
error. This almost always means the committed `artifacts/model.joblib`
is from a different scikit-learn version than what installed — make sure
you trained locally with the same `requirements.txt` that's in the repo
(don't `pip install` a different scikit-learn version by hand).

**Render service is very slow the first time you open it**
Expected — see the note in step 14. It only affects the first request
after 15+ minutes of no traffic.

**Port 8000 already in use, locally**
Something else on your PC is using it. Either stop that program, or run
uvicorn on a different port: add ` --port 8001` to the command in step 8
(and open `http://localhost:8001/` instead).

**Docker Desktop won't start / asks about WSL2**
Docker is optional (step 10) — you can skip it entirely and still deploy
to Render. If you do want it, follow Docker Desktop's own prompts to
enable WSL2 and restart your PC; this is a one-time setup issue unrelated
to this project.

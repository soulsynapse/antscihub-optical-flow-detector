# Notes for agents — working in this repo

How to *interact with* this project. Says nothing about what the project does
(it's changing too fast to document); see the code and `HANDOFF.md` for that.

## Running / driving the GUI

- Entry point: `python main.py` (PyQt6, `MainWindow(project_dir=".")`).
- **Use the repo venv: `.venv\Scripts\python.exe`.** It has PyQt6 (6.11.0) and
  `import gui.main_window` succeeds. The interpreters on the system PATH do NOT
  work: the anaconda python's PyQt6 is DLL-broken (`ImportError: DLL load failed
  while importing QtCore`), and Python311/Python37/WindowsApps have no PyQt6.
  A bare `python ...` will therefore fail — always call the venv python
  explicitly. (`.venv` is hidden/git-ignored, so a casual dir listing can miss it.)
- The GUI is driven **headlessly** with offscreen Qt — no display needed. Pattern
  (see `scripts/render_tabs.py`, `scripts/gui_smoke_test.py`):
  set `QT_QPA_PLATFORM=offscreen` (and `QT_QPA_FONTDIR=C:/Windows/Fonts` on
  Windows) *before* importing PyQt, build `MainWindow`, call `state.load_video(...)`
  / `state.open_cache(key)`, invoke slot methods directly, `app.processEvents()`,
  then `win.grab().save("screenshots/foo.png")` to eyeball a tab.
- **The smoke/render scripts are stale**: they call the *shelved* flat-AND tab3
  API (`tab3._sync_from_behavior`, `_rebuild_plots`, `tab3.plots`, `_add_constraint`).
  The live tab3 (`gui/tab3_behavior.py`) is the tree editor — `_sync_editor`,
  `_add_leaf`, a `QTreeWidget`, no `.plots` dict. Fix these calls before relying
  on those scripts, or write a fresh offscreen driver.

## Static checking without Qt

- Source files contain non-cp1252 glyphs (✏, ×, —, …). `ast.parse(open(f).read())`
  fails on Windows with a cp1252 `UnicodeDecodeError` — always pass
  `encoding='utf-8'`: `ast.parse(open(f, encoding='utf-8').read())`.
- `import gui.*` transitively imports PyQt6, so with a bare `python` it fails.
  With `.venv\Scripts\python.exe` it works, so prefer an actual
  `& .\.venv\Scripts\python.exe -c "import gui.main_window"` import check over a
  mere `ast.parse` when you want to catch more than syntax.

## Architecture touchpoints (stable enough to rely on)

- **`gui/state.py::AppState`** is the single shared hub. The three tabs never
  reference each other — they talk only through `AppState` signals
  (`video_loaded`, `cache_opened`, `rois_changed`, `frame_changed`,
  `behaviors_changed`, `status`, `request_tab`). To make one tab react to
  another's change, wire it through a state signal, not a direct call.
- **Per-video data lives in sidecar files next to the video**, via
  `AppState.video_sidecar(kind)` → `<video-path-without-ext>.<kind>.json`
  (returns `None` when no video loaded). Current kinds: `rois` (Tab 2 replicate
  boxes) and `marks` (Tab 3 manual ground-truth spans). Both auto-load on
  `video_loaded` and auto-save on edit. Root-level `marks.json` / `replicates.json`
  are legacy and no longer read.
- **Feature caches** are separate from that: keyed by
  `(video_hash, preprocessing, flow)` under `.cache/<key>/`, opened *by hand* in
  Tab 1 (nothing auto-opens a cache on video load). `load_video` drops the prior
  clip's cache so a new video starts uncached.

## Git conventions

- **Never `git commit` / `git push` without explicit permission** (repo + user
  CLAUDE.md). Ask first each time.
- Commit only the files you changed — the working tree carries unrelated churn
  (e.g. root `marks.json` gets rewritten by app runs; `behaviors/*.json` are
  user data). Stage explicit paths, never `git add -A`.
- Commit-message trailers are enforced by the harness (Co-Authored-By +
  Claude-Session); they're added automatically when you commit.

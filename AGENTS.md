# Notes for agents — working in this repo

How to *interact with* this project. Says nothing about what the project does
(it's changing too fast to document); see the code and `HANDOFF.md` for that.

## Running / driving the GUI

- Entry point: `python main.py` (PyQt6, `MainWindow(project_dir=".")`).
- **PyQt6 lives only in the anaconda python** (`~/AppData/Local/anaconda3`).
  The other interpreters on PATH (Python311, Python37, WindowsApps) don't have
  it — `where python` lists anaconda first.
- **As of this session the anaconda PyQt6 is broken**: any `import PyQt6.QtCore`
  fails with `ImportError: DLL load failed while importing QtCore: The specified
  procedure could not be found.` This is a Qt-DLL / binding version mismatch, not
  a PATH issue (adding `anaconda3\Library\bin` doesn't fix it). Until it's
  repaired you **cannot launch or headlessly drive the GUI here** — don't burn
  time retrying; fall back to static checks (below) and say so.
- When PyQt6 works, the GUI is driven **headlessly** with offscreen Qt — no
  display needed. Pattern (see `scripts/render_tabs.py`, `scripts/gui_smoke_test.py`):
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
- `import gui.*` transitively imports PyQt6, so it fails right now. Prefer
  `ast.parse` for syntax verification; you won't get an import-level check until
  PyQt6 is fixed.

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

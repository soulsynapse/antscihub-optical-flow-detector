# Notes for agents — working in this repo

How to *interact with* this project. For the current architecture and rationale,
see `README.md` and `docs/decisions.md`; historical handoffs are under
`docs/archive/` and are not current implementation contracts.

## Running / driving the GUI

- Entry point: `python main.py` (PyQt6, `MainWindow(project_dir=".")`).
- **Use the repo venv: `.venv\Scripts\python.exe`.** It has PyQt6 (6.11.0) and
  `import gui.main_window` succeeds. The interpreters on the system PATH do NOT
  work: the anaconda python's PyQt6 is DLL-broken (`ImportError: DLL load failed
  while importing QtCore`), and Python311/Python37/WindowsApps have no PyQt6.
  A bare `python ...` will therefore fail — always call the venv python
  explicitly. (`.venv` is hidden/git-ignored, so a casual dir listing can miss it.)
- The GUI is driven **headlessly** with offscreen Qt — no display needed. Pattern:
  set `QT_QPA_PLATFORM=offscreen` (and `QT_QPA_FONTDIR=C:/Windows/Fonts` on
  Windows) *before* importing PyQt, build `MainWindow`, call `state.load_video(...)`,
  invoke slot methods directly, `app.processEvents()`,
  then `win.grab().save("screenshots/foo.png")` to eyeball a tab.
- **There is no working offscreen driver right now.** The two that existed drove
  the retired flow-cache and Behavior tabs and were deleted with that subsystem.
  Write a fresh one against tabs 0-1 if you need it.

## Static checking without Qt

- Source files contain non-cp1252 glyphs (✏, ×, —, …). `ast.parse(open(f).read())`
  fails on Windows with a cp1252 `UnicodeDecodeError` — always pass
  `encoding='utf-8'`: `ast.parse(open(f, encoding='utf-8').read())`.
- `import gui.*` transitively imports PyQt6, so with a bare `python` it fails.
  With `.venv\Scripts\python.exe` it works, so prefer an actual
  `& .\.venv\Scripts\python.exe -c "import gui.main_window"` import check over a
  mere `ast.parse` when you want to catch more than syntax.

## Architecture touchpoints (stable enough to rely on)

- **`gui/state.py::AppState`** is the single shared hub. Tabs never
  reference each other — they talk only through `AppState` signals
  (`video_loaded`, `rois_changed`, `frame_changed`, `status`, `request_tab`,
  `calibration_changed`). To make one tab react to another's change, wire it
  through a state signal, not a direct call.
- **Tabs are `MainWindow.tabs` indices 0-1:** 0 Replicates (`tab2`), 1
  Preprocessing (live) (`tab_live`). The `tab2` field name predates the reorder —
  go by the index/label, not the number in the attribute. Neither tab is ever
  disabled once a video loads.
- **The flow cache is gone.** The flow-cache commit and Behavior Classification
  tabs, `core/cache.py`, `core/features.py`, `core/roi.py`, `AppState`'s cache
  API (`open_cache`/`has_cache`/`cache`/`ctx`/`roi_series`/…), and the
  cache-backed explorer shells were all deleted once the tensor path proved out.
  `AppState` is now video + config + replicate geometry only.
- **The tensor/scalogram detection path is the only path, and it is
  cache-independent.** `ScalogramExplorer` takes a
  `core.channel_source.ChannelData` from `live_channel_source` (a windowed live
  pass), not a raw cache. `core.tensor_channels.extract_channels_live`
  streams a windowed structure-tensor pass over a bare video;
  `core.wavelet.morlet_band_power` + `core.detection` (`detect_channel_region`,
  and the shared count/window/gate/clump functions the explorer also calls) run
  the detector; `gui/explorers/live_scalogram_surface.py` +
  `gui/tab_live_preprocess.py` host the live-tuning surface and the whole-video
  commit, and `gui/explorers/detection_timeline.py` is the navigation strip. When
  touching detection math, change `core/detection.py` — never fork a formula into
  the explorer, or the preview and the whole-clip pass will disagree.
- **Per-video data lives in sidecar files next to the video**, via
  `AppState.video_sidecar(kind)` → `<video-path-without-ext>.<kind>.json`
  (returns `None` when no video loaded). Current kinds: `rois` (Tab 1 replicate
  boxes) and `marks` (Tab 3 manual ground-truth spans). Both auto-load on
  `video_loaded` and auto-save on edit. Root-level `marks.json` / `replicates.json`
  are legacy and no longer read.

## Git conventions

- **Never `git commit` / `git push` without explicit permission** (repo + user
  CLAUDE.md). Ask first each time.
- Commit only the files you changed — the working tree carries unrelated churn
  (e.g. root `marks.json` gets rewritten by app runs; `behaviors/*.json` are
  user data). Stage explicit paths, never `git add -A`.
- Commit-message trailers are enforced by the harness (Co-Authored-By +
  Claude-Session); they're added automatically when you commit.

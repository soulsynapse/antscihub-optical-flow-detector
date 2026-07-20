# Shelved dev scripts

Both scripts drive the GUI through the flow-cache → Behavior-Classification
workflow, whose tabs were retired to `gui/_shelved/`. They reference
`win.tab1` / `win.tab3`, which `MainWindow` no longer builds, so both fail at
the first attribute access.

They were already stale before the retirement (they called the *first* flat-AND
tab3 API, not the tree editor that replaced it), so nothing here is a working
example — keep them only as a record of the offscreen-driver pattern.

The pattern itself is still correct and still worth copying for a fresh driver:
set `QT_QPA_PLATFORM=offscreen` (plus `QT_QPA_FONTDIR=C:/Windows/Fonts` on
Windows) *before* importing PyQt, build `MainWindow`, call
`state.load_video(...)`, invoke slot methods directly, `app.processEvents()`,
then `win.grab().save(...)`. A replacement should drive tabs 0-1 (Replicates,
live Preprocessing) and needs no cache.

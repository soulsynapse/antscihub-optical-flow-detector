"""Diagnostic explorer widgets and the shared plot-widget library.

These are Qt widgets, but they are diagnostic tools rather than production tabs,
so they live in their own subpackage beside the main-app tabs. ``plots.py`` holds
the reusable plot widgets (MiniPlot, DensityPlot, PixelBarPlot) and color
constants; ScalogramExplorer builds over a live ``ChannelData`` windowed source.
"""

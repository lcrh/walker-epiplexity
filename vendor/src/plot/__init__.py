"""Figure rendering for the experiments in ``src/``.

One module per experiment script, with the same filename: the experiment
computes and saves data, its plot module turns that data into figures. All
matplotlib code (layout, fonts, colors) lives here so the experiment scripts
read as pure experiment logic. Shared colors live in ``plot.palette``.
"""

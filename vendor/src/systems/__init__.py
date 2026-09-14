"""The systems being scored or trained: ECA rules, continuous-time flows, the
local NCA, and the MNIST encoder. Pure components (data generation and
``torch.nn`` models) with no experiment loops, CLI, or plotting.

These are the *scored* side of the scorer/scored boundary: nothing here may
import from ``rc_epiplexity`` (the frozen observer that evaluates them).
"""

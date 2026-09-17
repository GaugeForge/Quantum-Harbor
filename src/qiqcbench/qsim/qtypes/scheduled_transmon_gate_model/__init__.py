"""Scheduled native-gate transmon qtype.

The package models fixed, prevalidated ``rz``/``sx``/``x``/directed-``ecr``
schedules under edge-local, context-dependent noise.  It intentionally does
not expose an arbitrary circuit runner: task-enabled capabilities own the
finite experiment menus that can be executed on a particular device.
"""

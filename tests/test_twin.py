"""Physiology and twin sanity checks."""
import numpy as np
import pandas as pd

from glucotwin.data.meals import absorption_rate
from glucotwin.twin.engine import future_inputs, meal_habits, what_if
from glucotwin.twin.physiology import meal_appearance, params_from_clinical, simulate


def _p():
    return params_from_clinical(np.array([140.0]), np.array([12.0]), np.array([70.0]), np.array([8.0]))


def test_fasting_is_stable():
    G = simulate(_p(), 600)
    assert abs(G[0, -1] - 140) < 2


def test_meal_raises_glucose_and_returns():
    Ra = meal_appearance(600, [60], [70], [absorption_rate(70, 0.3)])
    G = simulate(_p(), 600, Ra=Ra)[0]
    assert G.max() > 190                      # clear post-meal spike for a T2D patient
    assert 40 < G.argmax() - 60 < 150         # peak 40-150 min after eating
    assert G[-1] < G.max() - 40               # coming back down


def test_low_gi_meal_gives_lower_peak():
    hi = simulate(_p(), 600, Ra=meal_appearance(600, [60], [50], [absorption_rate(85, 0.1)]))[0].max()
    lo = simulate(_p(), 600, Ra=meal_appearance(600, [60], [50], [absorption_rate(35, 0.5)]))[0].max()
    assert lo < hi


def test_sleep_debt_and_walk_effects():
    base = {"Ra": meal_appearance(300, [20], [70], [1 / 60]), "Ri": np.zeros(300), "A": np.zeros(300),
            "si": np.ones(300), "egp": np.ones(300)}
    G = what_if(_p(), (140.0, 0.0, 12.0), 600, 300, base,
                [{}, {"si_scale": 0.8}, {"walks": [(40, 25, 0.6)]}, {"egp_add": 0.12}])
    peak = G.max(1)
    assert peak[1] > peak[0]                  # poor sleep -> higher peak
    assert peak[2] < peak[0]                  # post-meal walk -> lower peak
    assert peak[3] > peak[0]                  # missed metformin -> higher


def test_forecast_inputs_use_only_past_meals():
    meals = pd.DataFrame({"minute": [100, 400], "carbs": [50.0, 60.0], "k": [1 / 60, 1 / 60], "slot": ["lunch", "dinner"]})
    ins = pd.DataFrame(columns=["minute", "dose", "ka"])
    Ra, _, _ = future_inputs(np.array([200]), 120, meals, ins, None, 0)
    assert Ra[0].sum() > 0                    # meal at 100 still absorbing
    Ra2, _, _ = future_inputs(np.array([300]), 120, meals[meals.minute == 400], ins, None, 0)
    assert Ra2.sum() == 0                     # future meal (400) is unknown at 300


def test_habits_learned_from_past_only():
    meals = pd.DataFrame({"minute": [480, 1440 + 480, 2 * 1440 + 900], "carbs": [40.0, 50.0, 99.0],
                          "k": [1 / 60] * 3, "slot": ["breakfast", "breakfast", "lunch"]})
    h = meal_habits(meals, 2)
    assert h["breakfast"][0] == 1.0 and h["lunch"][0] == 0.0

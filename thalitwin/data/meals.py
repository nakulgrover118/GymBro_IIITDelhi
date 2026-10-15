"""Indian meal library used by the simulator, the twin and the what-if tool.

Carbohydrate values are per typical home serving and glycaemic index (GI)
values are rounded from the International GI Tables (Atkinson et al., 2021)
and Indian studies (e.g. Mohan et al., ICMR-NIN food composition tables).
They are approximations meant for simulation, not dietary advice.

`slow` captures fat/protein/fibre that delays gastric emptying (0 = none,
1 = heavy fried / high-fat meal). Absorption speed in the twin depends on
both GI and `slow`.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Meal:
    key: str
    name: str
    carbs_g: float
    gi: int
    slow: float
    slot: str          # breakfast | lunch | dinner | snack
    region: str        # north | south | west | east | all
    veg: bool = True


MEALS = [
    # Breakfast
    Meal("idli_sambar", "Idli (3) + sambar", 48, 70, 0.2, "breakfast", "south"),
    Meal("masala_dosa", "Masala dosa + chutney", 58, 66, 0.4, "breakfast", "south"),
    Meal("pongal", "Ven pongal", 52, 68, 0.4, "breakfast", "south"),
    Meal("upma", "Rava upma", 42, 66, 0.3, "breakfast", "all"),
    Meal("poha", "Kanda poha", 46, 65, 0.3, "breakfast", "west"),
    Meal("aloo_paratha", "Aloo paratha (2) + curd", 62, 60, 0.6, "breakfast", "north"),
    Meal("besan_chilla", "Besan chilla (2)", 26, 40, 0.4, "breakfast", "north"),
    Meal("oats", "Masala oats", 34, 55, 0.3, "breakfast", "all"),
    Meal("bread_omelette", "Bread (2) + omelette", 30, 70, 0.4, "breakfast", "all", veg=False),
    Meal("ragi_dosa", "Ragi dosa (2)", 36, 50, 0.3, "breakfast", "south"),
    # Lunch / dinner
    Meal("rice_dal_sabzi", "Rice + dal + sabzi", 78, 70, 0.3, "lunch", "all"),
    Meal("roti_dal_sabzi", "Roti (3) + dal + sabzi", 62, 55, 0.4, "lunch", "north"),
    Meal("rajma_chawal", "Rajma chawal", 82, 60, 0.4, "lunch", "north"),
    Meal("sambar_rice", "Sambar rice + poriyal", 80, 72, 0.3, "lunch", "south"),
    Meal("curd_rice", "Curd rice", 62, 70, 0.3, "lunch", "south"),
    Meal("fish_curry_rice", "Fish curry + rice", 70, 70, 0.4, "lunch", "east", veg=False),
    Meal("veg_biryani", "Veg biryani + raita", 86, 68, 0.6, "dinner", "all"),
    Meal("chicken_biryani", "Chicken biryani + raita", 76, 65, 0.7, "dinner", "all", veg=False),
    Meal("chole_bhature", "Chole bhature", 88, 70, 0.9, "lunch", "north"),
    Meal("khichdi", "Moong dal khichdi", 60, 62, 0.3, "dinner", "all"),
    Meal("paneer_roti", "Paneer sabzi + roti (2)", 45, 50, 0.6, "dinner", "north"),
    Meal("ragi_mudde", "Ragi mudde + saaru", 56, 55, 0.3, "dinner", "south"),
    Meal("millet_thali", "Millet roti + dal + salad", 48, 48, 0.4, "dinner", "all"),
    Meal("brown_rice_dal", "Brown rice + dal + salad", 62, 50, 0.4, "dinner", "all"),
    Meal("chicken_salad", "Grilled chicken + salad", 14, 35, 0.6, "dinner", "all", veg=False),
    # Snacks / drinks / sweets
    Meal("sweet_chai", "Chai with 2 tsp sugar", 14, 75, 0.1, "snack", "all"),
    Meal("chai_biscuits", "Chai + 4 biscuits", 32, 72, 0.3, "snack", "all"),
    Meal("samosa", "Samosa (1) + chai", 42, 70, 0.8, "snack", "north"),
    Meal("banana", "Banana (1)", 27, 52, 0.1, "snack", "all"),
    Meal("mango", "Mango (1 cup)", 28, 56, 0.1, "snack", "all"),
    Meal("sprouts_chaat", "Sprouts chaat", 20, 35, 0.3, "snack", "all"),
    Meal("roasted_chana", "Roasted chana (30 g)", 15, 30, 0.4, "snack", "all"),
    Meal("gulab_jamun", "Gulab jamun (2)", 46, 80, 0.5, "snack", "all"),
    Meal("jalebi", "Jalebi (100 g)", 52, 85, 0.6, "snack", "north"),
    Meal("mithai_box", "Festival mithai (3 pcs)", 60, 78, 0.6, "snack", "all"),
]

MEAL_BY_KEY = {m.key: m for m in MEALS}


def absorption_rate(gi: float, slow: float) -> float:
    """Rate constant (1/min) of the two-compartment gut model.

    The carbohydrate appearance curve peaks at 1/k minutes after eating.
    High-GI, low-fat meals peak at ~35 min; heavy fried or low-GI meals at
    ~90+ min.
    """
    peak_min = 30.0 + 0.9 * (100.0 - gi) + 40.0 * slow
    return 1.0 / peak_min


def meals_for(slot: str, region: str):
    """Meals that fit a time slot and a regional diet pattern."""
    out = [m for m in MEALS if m.slot == slot and m.region in (region, "all")]
    if slot == "lunch":  # lunch and dinner menus are interchangeable at home
        out += [m for m in MEALS if m.slot == "dinner" and m.region in (region, "all")]
    if slot == "dinner":
        out += [m for m in MEALS if m.slot == "lunch" and m.region in (region, "all")]
    return out

# fuzzy logic inference engine
# adjusts knn career scores based on student stress and burnout levels
# uses scikit-fuzzy to implement the fuzzy inference system

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_fuzzy_system():
    """
    build and return the fuzzy inference system

    inputs:
        stress_level    numeric 1-10
        burnout_level   numeric 0-2 (low, medium, high)

    output:
        adjustment      numeric -30 to +10
                        negative means reduce suitability score
                        positive means boost suitability score
    """

    # define input and output universes
    stress = ctrl.Antecedent(np.arange(1, 11, 1), "stress")
    burnout = ctrl.Antecedent(np.arange(0, 3, 1), "burnout")
    adjustment = ctrl.Consequent(np.arange(-30, 11, 1), "adjustment")

    # stress membership functions
    stress["low"] = fuzz.trimf(stress.universe, [1, 1, 4])
    stress["medium"] = fuzz.trimf(stress.universe, [3, 5, 7])
    stress["high"] = fuzz.trimf(stress.universe, [6, 10, 10])

    # burnout membership functions
    burnout["low"] = fuzz.trimf(burnout.universe, [0, 0, 1])
    burnout["medium"] = fuzz.trimf(burnout.universe, [0, 1, 2])
    burnout["high"] = fuzz.trimf(burnout.universe, [1, 2, 2])

    # adjustment membership functions
    adjustment["strong_reduction"] = fuzz.trimf(adjustment.universe, [-30, -30, -15])
    adjustment["moderate_reduction"] = fuzz.trimf(adjustment.universe, [-20, -10, 0])
    adjustment["neutral"] = fuzz.trimf(adjustment.universe, [-5, 0, 5])
    adjustment["boost"] = fuzz.trimf(adjustment.universe, [0, 10, 10])

    # inference rules
    # if stress is high and burnout is high, strongly reduce score
    rule1 = ctrl.Rule(stress["high"] & burnout["high"], adjustment["strong_reduction"])

    # if stress is high and burnout is medium, moderately reduce score
    rule2 = ctrl.Rule(stress["high"] & burnout["medium"], adjustment["moderate_reduction"])

    # if stress is medium and burnout is high, moderately reduce score
    rule3 = ctrl.Rule(stress["medium"] & burnout["high"], adjustment["moderate_reduction"])

    # if stress is medium and burnout is medium, apply neutral adjustment
    rule4 = ctrl.Rule(stress["medium"] & burnout["medium"], adjustment["neutral"])

    # if stress is low and burnout is low, boost score
    rule5 = ctrl.Rule(stress["low"] & burnout["low"], adjustment["boost"])

    # if stress is low and burnout is medium, apply neutral adjustment
    rule6 = ctrl.Rule(stress["low"] & burnout["medium"], adjustment["neutral"])

    # if stress is medium and burnout is low, apply neutral adjustment
    rule7 = ctrl.Rule(stress["medium"] & burnout["low"], adjustment["neutral"])

    # if stress is high and burnout is low, moderately reduce score
    rule8 = ctrl.Rule(stress["high"] & burnout["low"], adjustment["moderate_reduction"])

    # if stress is low but burnout is high, still moderately reduce score
    # (burnout is a longer-term wellbeing signal - a low CURRENT stress
    # reading does not cancel out an already-high burnout state; this
    # rule fixes a gap in the original rule base where this combination
    # was never covered, causing the fuzzy system to fail to compute an
    # output entirely for these inputs)
    rule9 = ctrl.Rule(stress["low"] & burnout["high"], adjustment["moderate_reduction"])

    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    simulation = ctrl.ControlSystemSimulation(system)

    return simulation


def get_adjustment(simulation, stress_level: float, burnout_level: int) -> float:
    """
    compute the fuzzy adjustment value for a given stress and burnout input

    args:
        simulation:     the fuzzy control system simulation object
        stress_level:   student stress score (1-10)
        burnout_level:  encoded burnout level (0=low, 1=medium, 2=high)

    returns:
        float adjustment value between -30 and +10
    """

    simulation.input["stress"] = float(stress_level)
    simulation.input["burnout"] = float(burnout_level)
    simulation.compute()

    return round(simulation.output["adjustment"], 2)


def adjust_scores(scores: dict, stress_level: float, burnout_level: int) -> tuple:
    """
    apply fuzzy adjustment to all knn career scores

    args:
        scores:         dict of {career_cluster: knn_score} from knn prediction
        stress_level:   student stress score (1-10)
        burnout_level:  encoded burnout level (0=low, 1=medium, 2=high)

    returns:
        tuple of (adjusted_scores dict, adjustment value, wellbeing_summary string)
    """

    simulation = build_fuzzy_system()
    adjustment = get_adjustment(simulation, stress_level, burnout_level)

    adjusted_scores = {}
    for career, score in scores.items():
        adjusted = score + adjustment
        # clamp to 0-100 range
        adjusted_scores[career] = round(max(0.0, min(100.0, adjusted)), 2)

    # generate wellbeing summary based on adjustment
    if adjustment <= -20:
        summary = "your stress and burnout levels are high. we have adjusted recommendations toward lower-pressure career paths. consider speaking with a counsellor."
    elif adjustment <= -5:
        summary = "moderate stress detected. recommendations have been slightly adjusted. prioritising your wellbeing is important."
    elif adjustment >= 8:
        summary = "you appear energised and motivated. all career paths are well within reach."
    else:
        summary = None

    return adjusted_scores, adjustment, summary


def get_wellbeing_flag(adjustment: float) -> str | None:
    """return a short wellbeing flag string based on the adjustment value"""

    if adjustment <= -20:
        return "high stress alert"
    elif adjustment <= -5:
        return "moderate stress detected"
    else:
        return None
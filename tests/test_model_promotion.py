from src.model_promotion import is_candidate_better, build_cd_decision


def test_promotes_when_ap_and_precision_improve_and_recall_drop_is_acceptable():
    champion = {
        "average_precision": 0.80,
        "production_recall": 0.70,
        "production_precision": 0.40,
    }

    challenger = {
        "average_precision": 0.82,
        "production_recall": 0.69,
        "production_precision": 0.42,
    }

    promoted, reason = is_candidate_better(champion, challenger)

    assert promoted is True
    assert "promu" in reason.lower()


def test_rejects_when_average_precision_gain_is_insufficient():
    champion = {
        "average_precision": 0.80,
        "production_recall": 0.70,
        "production_precision": 0.40,
    }

    challenger = {
        "average_precision": 0.805,
        "production_recall": 0.70,
        "production_precision": 0.42,
    }

    promoted, reason = is_candidate_better(champion, challenger)

    assert promoted is False
    assert "average_precision" in reason


def test_rejects_when_recall_drop_is_too_high():
    champion = {
        "average_precision": 0.80,
        "production_recall": 0.70,
        "production_precision": 0.40,
    }

    challenger = {
        "average_precision": 0.82,
        "production_recall": 0.67,
        "production_precision": 0.42,
    }

    promoted, reason = is_candidate_better(champion, challenger)

    assert promoted is False
    assert "recall" in reason.lower()


def test_rejects_when_precision_gain_is_insufficient():
    champion = {
        "average_precision": 0.80,
        "production_recall": 0.70,
        "production_precision": 0.40,
    }

    challenger = {
        "average_precision": 0.82,
        "production_recall": 0.70,
        "production_precision": 0.405,
    }

    promoted, reason = is_candidate_better(champion, challenger)

    assert promoted is False
    assert "precision" in reason.lower()


def test_build_cd_decision_contains_champion_and_challenger_precision():
    champion_info = {
        "version": "12",
        "metrics": {
            "average_precision": 0.80,
            "production_recall": 0.70,
            "production_precision": 0.40,
        },
    }

    challenger_info = {
        "version": "13",
        "metrics": {
            "average_precision": 0.82,
            "production_recall": 0.69,
            "production_precision": 0.42,
        },
    }

    decision = build_cd_decision(
        model_name="fraud_model",
        champion_info=champion_info,
        challenger_info=challenger_info,
        promoted=True,
        decision_reason="Challenger promu : critères de promotion satisfaits.",
    )

    assert decision["champion_precision_fraud"] == 0.40
    assert decision["challenger_precision_fraud"] == 0.42
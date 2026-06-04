from src.model_promotion import is_candidate_better


def test_candidate_promoted_when_metrics_improve_enough():
    champion_metrics = {
        "average_precision": 0.80,
        "production_recall": 0.70,
    }

    challenger_metrics = {
        "average_precision": 0.82,
        "production_recall": 0.72,
        "production_precision": 0.61,
    }

    promoted, reason = is_candidate_better(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
    )

    assert promoted is True
    assert isinstance(reason, str)


def test_candidate_not_promoted_when_average_precision_gain_too_low():
    champion_metrics = {
        "average_precision": 0.8696,
        "production_recall": 0.8554,
    }

    challenger_metrics = {
        "average_precision": 0.8779,
        "production_recall": 0.8671,
        "production_precision": 0.66,
    }

    promoted, reason = is_candidate_better(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
    )

    assert promoted is False
    assert "average_precision" in reason


def test_candidate_not_promoted_when_recall_drops_too_much():
    champion_metrics = {
        "average_precision": 0.80,
        "production_recall": 0.80,
    }

    challenger_metrics = {
        "average_precision": 0.83,
        "production_recall": 0.75,
        "production_precision": 0.60,
    }

    promoted, reason = is_candidate_better(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
    )

    assert promoted is False
    assert "recall" in reason.lower()


def test_candidate_not_promoted_when_precision_too_low():
    champion_metrics = {
        "average_precision": 0.80,
        "production_recall": 0.70,
    }

    challenger_metrics = {
        "average_precision": 0.83,
        "production_recall": 0.72,
        "production_precision": 0.05,
    }

    promoted, reason = is_candidate_better(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
    )

    assert promoted is False
    assert "precision" in reason.lower()
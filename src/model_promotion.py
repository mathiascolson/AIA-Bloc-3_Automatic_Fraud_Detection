from __future__ import annotations


def is_candidate_better(
    champion_metrics: dict,
    challenger_metrics: dict,
    min_average_precision_gain: float = 0.01,
    max_recall_drop: float = 0.02,
    min_precision_fraud: float = 0.10,
) -> tuple[bool, str]:
    """
    Détermine si le challenger doit être promu champion.

    Règle :
    - le challenger doit améliorer average_precision d'au moins min_average_precision_gain ;
    - le recall au seuil de production ne doit pas trop baisser ;
    - la precision au seuil de production doit rester acceptable.
    """

    required_champion_metrics = [
        "average_precision",
        "production_recall",
    ]

    required_challenger_metrics = [
        "average_precision",
        "production_recall",
        "production_precision",
    ]

    missing_champion_metrics = [
        metric
        for metric in required_champion_metrics
        if metric not in champion_metrics
    ]

    missing_challenger_metrics = [
        metric
        for metric in required_challenger_metrics
        if metric not in challenger_metrics
    ]

    if missing_champion_metrics:
        raise ValueError(
            f"Métriques champion manquantes : {missing_champion_metrics}"
        )

    if missing_challenger_metrics:
        raise ValueError(
            f"Métriques challenger manquantes : {missing_challenger_metrics}"
        )

    champion_ap = champion_metrics["average_precision"]
    challenger_ap = challenger_metrics["average_precision"]

    champion_recall = champion_metrics["production_recall"]
    challenger_recall = challenger_metrics["production_recall"]

    challenger_precision = challenger_metrics["production_precision"]

    improves_average_precision = (
        challenger_ap >= champion_ap + min_average_precision_gain
    )

    recall_not_degraded_too_much = (
        challenger_recall >= champion_recall - max_recall_drop
    )

    precision_is_acceptable = (
        challenger_precision >= min_precision_fraud
    )

    if not improves_average_precision:
        return (
            False,
            (
                "Challenger non promu : amélioration insuffisante de "
                "average_precision."
            ),
        )

    if not recall_not_degraded_too_much:
        return (
            False,
            (
                "Challenger non promu : dégradation trop importante du "
                "recall fraude au seuil de production."
            ),
        )

    if not precision_is_acceptable:
        return (
            False,
            (
                "Challenger non promu : precision fraude inférieure au "
                "seuil minimal."
            ),
        )

    return (
        True,
        "Challenger promu : critères de promotion satisfaits.",
    )


def build_cd_decision(
    model_name: str,
    champion_info: dict,
    challenger_info: dict,
    promoted: bool,
    decision_reason: str,
) -> dict:
    """
    Construit le dictionnaire à insérer dans model_cd_decisions.
    """

    champion_metrics = champion_info["metrics"]
    challenger_metrics = challenger_info["metrics"]

    return {
        "model_name": model_name,
        "champion_version": str(champion_info["version"]),
        "challenger_version": str(challenger_info["version"]),
        "champion_average_precision": champion_metrics["average_precision"],
        "challenger_average_precision": challenger_metrics["average_precision"],
        "champion_recall_fraud": champion_metrics["production_recall"],
        "challenger_recall_fraud": challenger_metrics["production_recall"],
        "challenger_precision_fraud": challenger_metrics["production_precision"],
        "promoted": promoted,
        "decision_reason": decision_reason,
    }
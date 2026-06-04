from __future__ import annotations

import uuid
from dataclasses import dataclass

import pandas as pd


class DataQualityError(ValueError):
    """
    Erreur levée quand un contrôle de qualité de données échoue.
    """


@dataclass
class DataQualityCheckResult:
    expectation_name: str
    success: bool
    details: str


INCOMING_TRANSACTION_REQUIRED_COLUMNS = [
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "merch_lat",
    "merch_long",
    "is_fraud",
    "current_time",
]


def _build_dataframe_batch(df: pd.DataFrame, asset_name: str):
    """
    Construit un batch Great Expectations à partir d'un DataFrame pandas.
    """

    import great_expectations as gx

    context = gx.get_context()

    unique_suffix = uuid.uuid4().hex[:12]

    data_source = context.data_sources.add_pandas(
        name=f"pandas_{asset_name}_{unique_suffix}"
    )

    data_asset = data_source.add_dataframe_asset(
        name=f"{asset_name}_{unique_suffix}"
    )

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name=f"batch_{asset_name}_{unique_suffix}"
    )

    batch = batch_definition.get_batch(
        batch_parameters={
            "dataframe": df,
        }
    )

    return batch


def _validation_success(validation_result) -> bool:
    """
    Récupère le booléen success d'un résultat GX.
    """

    if hasattr(validation_result, "success"):
        return bool(validation_result.success)

    if isinstance(validation_result, dict):
        return bool(validation_result.get("success"))

    try:
        return bool(validation_result["success"])
    except Exception:
        return False


def _validation_details(validation_result) -> str:
    """
    Produit un résumé court du résultat GX.
    """

    if hasattr(validation_result, "to_json_dict"):
        result_dict = validation_result.to_json_dict()
        return str(result_dict.get("result", result_dict))

    if isinstance(validation_result, dict):
        return str(validation_result.get("result", validation_result))

    return str(validation_result)


def _run_expectations(batch, expectations: list) -> list[DataQualityCheckResult]:
    """
    Exécute une liste d'Expectations GX.
    """

    results: list[DataQualityCheckResult] = []

    for expectation in expectations:
        validation_result = batch.validate(expectation)

        results.append(
            DataQualityCheckResult(
                expectation_name=expectation.__class__.__name__,
                success=_validation_success(validation_result),
                details=_validation_details(validation_result),
            )
        )

    return results


def _raise_if_failed(
    results: list[DataQualityCheckResult],
    context: str,
) -> None:
    """
    Lève une erreur si au moins un contrôle échoue.
    """

    failed_results = [
        result
        for result in results
        if not result.success
    ]

    if not failed_results:
        print(f"[GX] Validation réussie : {context}")
        return

    formatted_failures = "\n".join(
        [
            f"- {result.expectation_name}: {result.details}"
            for result in failed_results
        ]
    )

    raise DataQualityError(
        f"[GX] Validation échouée : {context}\n{formatted_failures}"
    )


def validate_incoming_transactions(df: pd.DataFrame) -> None:
    """
    Valide les transactions entrantes avant prédiction et stockage.

    Cette validation bloque :
    - les transactions incomplètes ;
    - les montants négatifs ;
    - les coordonnées incohérentes ;
    - les labels is_fraud invalides ;
    - les doublons de trans_num dans le batch.
    """

    if df.empty:
        raise DataQualityError("[GX] Le batch de transactions entrantes est vide.")

    import great_expectations as gx

    batch = _build_dataframe_batch(
        df=df,
        asset_name="incoming_transactions",
    )

    expectations = [
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=INCOMING_TRANSACTION_REQUIRED_COLUMNS,
            exact_match=False,
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="trans_num",
        ),
        gx.expectations.ExpectColumnValuesToBeUnique(
            column="trans_num",
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="cc_num",
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="merchant",
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="category",
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="amt",
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="amt",
            min_value=0,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="lat",
            min_value=-90,
            max_value=90,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="long",
            min_value=-180,
            max_value=180,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="merch_lat",
            min_value=-90,
            max_value=90,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="merch_long",
            min_value=-180,
            max_value=180,
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="current_time",
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="is_fraud",
            value_set=[0, 1],
        ),
    ]

    results = _run_expectations(
        batch=batch,
        expectations=expectations,
    )

    _raise_if_failed(
        results=results,
        context="incoming_transactions",
    )
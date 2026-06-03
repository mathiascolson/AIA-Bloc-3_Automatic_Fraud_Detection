from __future__ import annotations

from io import StringIO
import time

import pandas as pd
import requests


def get_with_retries(
    url: str,
    timeout: int = 30,
    retries: int = 3,
    delay_seconds: int = 5,
) -> requests.Response:
    """
    Appel GET avec retries simples pour limiter les échecs réseau transitoires.
    """

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            print(f"[API] Tentative {attempt}/{retries} échouée : {exc}")

            if attempt < retries:
                time.sleep(delay_seconds)

    raise RuntimeError(
        f"Échec de l'appel GET après {retries} tentatives : {url}"
    ) from last_error


def parse_transactions_response(data) -> pd.DataFrame:
    """
    Convertit la réponse API en DataFrame.

    Gère plusieurs formats possibles :
    - chaîne JSON au format orient='split' ;
    - chaîne JSON classique ;
    - dictionnaire orient='split' ;
    - liste de dictionnaires ;
    - dictionnaire simple convertible en DataFrame.
    """

    if isinstance(data, str):
        try:
            return pd.read_json(StringIO(data), orient="split")
        except ValueError:
            return pd.read_json(StringIO(data))

    if isinstance(data, dict) and {"columns", "data", "index"}.issubset(data.keys()):
        return pd.DataFrame(
            data=data["data"],
            columns=data["columns"],
            index=data["index"],
        )

    if isinstance(data, list):
        return pd.DataFrame(data)

    return pd.DataFrame(data)


def fetch_current_transactions(api_url: str) -> pd.DataFrame:
    """
    Récupère les transactions depuis l'API externe Jedha.
    """

    response = get_with_retries(api_url)
    data = response.json()
    df = parse_transactions_response(data)

    print(f"[API] Transactions récupérées : {len(df)}")

    return df
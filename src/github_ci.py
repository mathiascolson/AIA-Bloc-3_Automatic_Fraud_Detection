from __future__ import annotations

import requests


def get_latest_workflow_run(
    owner: str,
    repo: str,
    workflow_name: str,
    branch: str = "main",
    github_token: str | None = None,
) -> dict:
    """
    Récupère le dernier run GitHub Actions pour un workflow donné.
    """

    url = (
        f"https://api.github.com/repos/{owner}/{repo}/actions/"
        f"workflows/{workflow_name}/runs"
    )

    headers = {
        "Accept": "application/vnd.github+json",
    }

    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    params = {
        "branch": branch,
        "per_page": 1,
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()
    workflow_runs = payload.get("workflow_runs", [])

    if not workflow_runs:
        raise ValueError(
            f"Aucun run GitHub Actions trouvé pour {workflow_name} "
            f"sur la branche {branch}."
        )

    return workflow_runs[0]


def is_latest_workflow_successful(
    owner: str,
    repo: str,
    workflow_name: str,
    branch: str = "main",
    github_token: str | None = None,
) -> tuple[bool, dict]:
    """
    Vérifie si le dernier run du workflow GitHub Actions est terminé en succès.
    """

    latest_run = get_latest_workflow_run(
        owner=owner,
        repo=repo,
        workflow_name=workflow_name,
        branch=branch,
        github_token=github_token,
    )

    status = latest_run.get("status")
    conclusion = latest_run.get("conclusion")

    is_successful = (
        status == "completed"
        and conclusion == "success"
    )

    return is_successful, latest_run
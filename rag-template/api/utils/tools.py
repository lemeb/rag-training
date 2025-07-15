from typing import Any, cast

import requests
from duckduckgo_search import DDGS
from exa_py import Exa
import os


def duckduckgo_search(query: str) -> list[dict[str, str]]:
    return DDGS().text(query, max_results=20)


def exa_search(query: str) -> list[dict[str, str]]:
    return cast(
        list[dict[str, str]],
        Exa(api_key=os.getenv("EXA_API_KEY"))
        .search_and_contents(query, text=True, type="keyword")
        .results,
    )


def get_current_weather(latitude: float, longitude: float) -> dict[str, Any] | None:  # pyright: ignore[reportExplicitAny]
    # Format the URL with proper parameter substitution
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m&hourly=temperature_2m&daily=sunrise,sunset&timezone=auto"

    try:
        # Make the API call
        response = requests.get(url)

        # Raise an exception for bad status codes
        response.raise_for_status()

        # Return the JSON response
        return response.json()  # pyright: ignore[reportAny]

    except requests.RequestException as e:
        # Handle any errors that occur during the request
        print(f"Error fetching weather data: {e}")
        return None

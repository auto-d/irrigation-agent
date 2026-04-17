# Weather Service

This directory contains the in-repo National Weather Service forecast integration.

Planner code should prefer importing the module directly instead of shelling out to an external tool.

## Supported Inputs

- latitude and longitude

The NWS API's native forecast lookup path is a point lookup followed by the forecast URLs returned for that point. City and state are not currently supported here because that requires an additional geocoding provider.

## Direct Python Usage

```python
import asyncio

from services.weather import WeatherClient


async def main() -> None:
    async with WeatherClient.from_env() as client:
        forecast = await client.fetch_forecast(42.8864, -78.8784)
        hourly = await client.fetch_hourly_forecast(42.8864, -78.8784)
        print(forecast.periods[0].short_forecast)
        print(hourly.periods[0].temperature)


asyncio.run(main())
```

## Configuration

- `NWS_USER_AGENT`: meaningful user agent string for api.weather.gov requests
- `NWS_API_BASE_URL`: override for testing if needed

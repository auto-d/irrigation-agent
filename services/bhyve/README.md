# B-hyve Service

This directory contains the in-repo B-hyve integration used by the agent project.

The goal is to keep the planner-facing interface small and stable while hiding the quirks of the Orbit transport layer.

For planner code, prefer importing the module directly instead of shelling out to the CLI.

## Layout

- `controller.py`: high-level control wrapper with `on`, `off`, and `cycle`
- `tools/probe.py`: credential and device-inspection probe
- `tools/explore.py`: richer API exploration helper
- `vendor/pybhyve/`: vendored minimal upstream client dependency derived from `sebr/pybhyve`

## Behavior Notes

- The HT25 valve has been manually verified to actuate through this wrapper.
- Orbit websocket acknowledgements are not reliable enough to use as the sole success criterion.
- `--poll` can be used to poll REST status after a command, but this is secondary evidence only.
- The planner-facing commands are `on`, `off`, and `cycle`; lower-level transport details are intentionally not exposed.

## CLI Usage

Run from the repository root:

```bash
python -m services.bhyve.controller devices
python -m services.bhyve.controller status
python -m services.bhyve.controller on --seconds 60
python -m services.bhyve.controller off
python -m services.bhyve.controller cycle --seconds 5
```

Optional verification flags:

```bash
python -m services.bhyve.controller on --seconds 60 --poll
python -m services.bhyve.controller cycle --seconds 5 --poll
```

Examples:

```bash
python -m services.bhyve.controller on --seconds 120
python -m services.bhyve.controller off
python -m services.bhyve.controller cycle --seconds 10
```

What those mean:

- `on --seconds N`: open the valve now and provision enough B-hyve runtime budget for at least `N` seconds
- `off`: close the valve now
- `cycle --seconds N`: open the valve, wait `N` seconds in the client, then send `off`

## Credentials

The controller expects:

- `BHYVE_EMAIL`
- `BHYVE_PASSWORD`

Keep credentials in a local `.env` or shell session only; do not commit them.

## Helper Tools

The probe and exploration helpers remain useful during debugging:

```bash
python -m services.bhyve.tools.probe
python -m services.bhyve.tools.explore
```

They are not the intended planner-facing interface; use `services.bhyve.controller` for that.

## Direct Python Usage

Planner code should use the module directly:

```python
import asyncio

from services.bhyve.controller import controller_from_env


async def main() -> None:
    async with controller_from_env() as controller:
        device_id = controller.list_sprinkler_devices()[0]["id"]
        await controller.turn_on(device_id, seconds=30)
        await controller.turn_off(device_id)
        await controller.cycle(device_id, seconds=5)


asyncio.run(main())
```

Recommended async methods:

- `controller.list_sprinkler_devices()`
- `await controller.turn_on(device_id, seconds=...)`
- `await controller.turn_off(device_id)`
- `await controller.cycle(device_id, seconds=...)`
- `await controller.get_history(device_id)`
- `await controller.get_landscapes(device_id)`

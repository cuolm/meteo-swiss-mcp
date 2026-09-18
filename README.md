# MeteoSwiss MCP Server
[![PyPI](https://img.shields.io/pypi/v/meteo-swiss-mcp.svg)](https://pypi.org/project/meteo-swiss-mcp/)
[![License](https://img.shields.io/github/license/cuolm/meteo-swiss-mcp.svg)](/LICENSE.txt)
[![Release](https://github.com/cuolm/meteo-swiss-mcp/actions/workflows/release.yaml/badge.svg)](https://github.com/cuolm/meteo-swiss-mcp/actions/workflows/release.yaml)
[![Tests](https://github.com/cuolm/meteo-swiss-mcp/actions/workflows/tests.yaml/badge.svg)](https://github.com/cuolm/meteo-swiss-mcp/actions/workflows/tests.yaml)

A **Model Context Protocol ([MCP](https://modelcontextprotocol.info/))** server that exposes Swiss weather forecast data as callable tools.
It fetches data from the official [MeteoSwiss](https://opendatadocs.meteoswiss.ch/e-forecast-data/e2-e3-numerical-weather-forecasting-model) [meteodata-lab](https://meteoswiss.github.io/meteodata-lab/), caches it locally, and serves predictions such as rainfall, sunshine, temperature, etc. The prediction data is from the [ICON-CH2-EPS](https://www.meteoswiss.admin.ch/weather/warning-and-forecasting-systems/icon-forecasting-systems.html) forecast system that produces data for up to 5 days ahead.

Additionally there is also an MCP client that can be run to test the server using the stdio transport.

**Note:**
This project is **not an official MeteoSwiss product**.
All forecast data are from the [MeteoSwiss Open Data](https://opendata.swiss/en/organization/bundesamt-fur-meteorologie-und-klimatologie-meteoschweiz) portal.
**Source: MeteoSwiss**

## Table of Contents
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Available Tools](#available-tools)
- [Example Usage with LMStudio](#example-usage-with-lmstudio)
- [Tests](#tests)
- [Releasing](#releasing)
- [Resources](#resources)
- [License](#license)

## Project Structure
```text
meteo-swiss-mcp/
├── src/meteo_swiss_mcp/
│   ├── server.py           # MCP server
│   ├── predictions.py      # Data fetching logic
│   ├── client.py           # MCP client (optional)
│   └── log_config.json     # Packaged logging configuration
├── tests/meteo_swiss_mcp/  # Pytest suite
├── .github/workflows/      # CI and release pipelines
├── docs/                   # Documentation
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Pinned, reproducible dependency set
├── .env                    # NOMINATIM_USER_AGENT (not committed)
└── Dockerfile
```
Caches live outside the project, under your OS's standard cache directory (see [Installation](#installation)).

## Quick Start

### 1. Installation
Install the server globally to run it anywhere on your system:
```bash
uv tool install meteo-swiss-mcp
```

### 2. Configuration
Create a `.env` file with your Nominatim user agent (see [Configuration](#configuration)):
```bash
echo 'NOMINATIM_USER_AGENT="YourWeatherMCPServer/1.0 (yourname@example.com)"' > .env
```

### 3. Execution
Run the server from the directory containing your `.env` file:
```bash
meteo-swiss-mcp-server
```

## Installation

### As a Global CLI Tool
```bash
uv tool install meteo-swiss-mcp
```

### As a Library Dependency
```bash
# Using uv
uv add meteo-swiss-mcp

# Using pip
pip install meteo-swiss-mcp
```

> **Note:** Add the `client` extra (`meteo-swiss-mcp[client]`) if you also want the optional MCP client, which pulls in the Ollama SDK.

### From Source
```bash
git clone https://github.com/cuolm/meteo-swiss-mcp.git
cd meteo-swiss-mcp

# Using uv (Recommended)
uv sync --extra client

# Using pip
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[client]"
```

> **Note:** `uv sync` on its own installs the server only. The `--extra client` flag is
> what pulls in the Ollama SDK needed by `meteo-swiss-mcp-client`.

**Note:**
- [Ollama](https://ollama.com/) is optional – only needed if you want to use the MCP client (`meteo-swiss-mcp-client`, installed via the `client` extra).
- The server reads its `.env` file relative to the **current working directory** — run it from the directory that holds your `.env` file (or export the variable directly).
- Caches are stored under your OS's standard cache directory (via [platformdirs](https://github.com/tox-dev/platformdirs), e.g. `~/Library/Caches/meteo-swiss-mcp` on macOS, `~/.cache/meteo-swiss-mcp` on Linux) — independent of where the server is launched from, so downloaded forecasts and geocoded locations are reused across runs.
  - `EarthKitCache/` avoids re‑downloading weather data. Delete it to clear.
  - `nominatim_geocode_cache.json` caches lat/lon lookups. Delete it to clear.

## Configuration

Create a `.env` file in the directory you'll run the server from, specifying an environment variable that tells Nominatim (the geocoding service) who is making the call.

```bash
echo 'NOMINATIM_USER_AGENT="YourWeatherMCPServer/1.0 (yourname@example.com)"' > .env
```

> **Note:** Replace the application name and address with your own. The
> [Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/) requires a
> user agent identifying a real application and contact address, and blocks requests that do not
> provide one. Keep lookups to at most one per second; results are cached, so only locations that
> have not been requested before reach the service.

## Usage

### Running the Server
If installed via `uv tool install` or `pip`:
```bash
# stdio (default)
meteo-swiss-mcp-server

# streamable-http
meteo-swiss-mcp-server --transport=streamable-http --host=localhost --port=8050
```

If running within the source repository cloned from GitHub:
```bash
# Using uv (Recommended)
uv run meteo-swiss-mcp-server

# Using pip, with the virtual environment activated
meteo-swiss-mcp-server
```
Optional flags: `--help`

### Running the Server with Docker
Images are built and published automatically by GitHub Actions to the project's [GitHub Container Registry](https://ghcr.io/cuolm/meteo-swiss-mcp), tagged `:latest` (newest release) and by version.

1. Create a `.env` file containing your Nominatim user agent environment variable (replace `"YourWeatherMCPServer/1.0 (yourname@example.com)"`):
```bash
echo 'NOMINATIM_USER_AGENT="YourWeatherMCPServer/1.0 (yourname@example.com)"' > .env
```
2. Run the published image, passing the `.env` file and mapping port 8050:
```bash
docker run --env-file .env -p 8050:8050 ghcr.io/cuolm/meteo-swiss-mcp:latest
```
3. Access the server at:
```bash
http://localhost:8050/mcp/
```

#### Manual Build
```bash
docker build -t meteo-swiss-mcp .
docker run --env-file .env -p 8050:8050 meteo-swiss-mcp
```

### Running the MCP Client using Stdio Transport
The bundled MCP client can be used to test the server over the stdio transport. It requires the `client` extra (see [Installation](#installation)).
Make sure Ollama is installed on your system. You can [download it here](https://ollama.com/download) or install via Homebrew on macOS: `brew install ollama`

```bash
# Pull a local Ollama LLM model (e.g. qwen3:4b)
ollama pull qwen3:4b

# Run the MCP client (it automatically starts the server as a subprocess)
meteo-swiss-mcp-client --model=qwen3:4b

# From a source checkout, using uv
uv run --extra client meteo-swiss-mcp-client --model=qwen3:4b
```

## Available Tools

| Tool | Purpose | Example Call |
|------|---------|--------------|
| `current_date_and_time()` | Current date and time (weekday day.month.year hour:minute:second) in Swiss local time | `current_date_and_time()` |
| `total_rainfall(location, lead_time_start_swiss, lead_time_end_swiss)` | Total rainfall (mm) for a period | `total_rainfall("Zurich", 24, 48)` |
| `sunshine_hours(location, lead_time_start_swiss, lead_time_end_swiss)` | Sunshine hours for a period | `sunshine_hours("Zurich", 24, 48)` |
| `temperature(location, lead_time_swiss)` | Max temperature (°C) at a specific lead time | `temperature("Zurich", 36)` |
| `wind_speed(location, lead_time_swiss)` | Wind speed (m/s) at a specific lead time | `wind_speed("Zurich", 36)` |
| `pressure_msl(location, lead_time_swiss)` | Sea‑level pressure (Pa) at a specific lead time | `pressure_msl("Zurich", 36)` |
| `total_cloud_cover(location, lead_time_swiss)` | Cloud cover (%) at a specific lead time | `total_cloud_cover("Zurich", 36)` |
| `snow_depth(location, lead_time_swiss)` | Snow depth (m) at a specific lead time | `snow_depth("Zurich", 36)` |
| `precipitation_rate(location, lead_time_swiss)` | Precipitation rate (mm/s) at a specific lead time | `precipitation_rate("Zurich", 36)` |

**Lead Time**
- Lead time is the number of hours counted from Swiss local time 00:00, internally converted to UTC (the ICON-CH2-EPS forecast system uses UTC).
- Example: A lead time of 36 hours returns the forecast for 12:00 Swiss local time tomorrow.
- Minimum lead time: 2 hours; maximum lead time: 121 hours.

## Example Usage with LMStudio

### Using the streamable-http transport layer
Configure the mcp.json file in [LMStudio](https://lmstudio.ai/):
```json
{
  "mcpServers": {
    "meteo_swiss_mcp_server": {
      "url": "http://localhost:8050/mcp/"
    }
  }
}
```
Run the MCP server with the streamable-http transport layer:
```bash
uv run meteo-swiss-mcp-server --transport=streamable-http --host=localhost --port=8050
```

### Using the stdio transport layer
Configure the mcp.json file in LMStudio. Replace `<path-to-the-project>` with your actual local path:
```json
{
  "mcpServers": {
    "meteo_swiss_mcp_server": {
      "command": "<path-to-the-project>/.venv/bin/meteo-swiss-mcp-server"
    }
  }
}
```
![LMStudioMCPServer](docs/LMStudioMCPServer.png)

## Tests
Run the test suite from the project root with:
```bash
uv run pytest

# Or, with an activated virtual environment
pytest
```

Every push and pull request runs the suite plus a Docker build check via the [Tests workflow](.github/workflows/tests.yaml).

## Releasing

Versions are derived from Git tags by `hatch-vcs` — there is no version string to bump by hand.

- Pushing a pre-release tag (e.g. `0.2.0rc1`) triggers [`release_test.yaml`](.github/workflows/release_test.yaml): tests, publish to **TestPyPI**, push a versioned image to GHCR, and create a prerelease GitHub Release.
- Pushing a final tag (e.g. `0.2.0`) triggers [`release.yaml`](.github/workflows/release.yaml): tests, publish to **PyPI**, push `:<version>` and `:latest` images to GHCR, and create a GitHub Release.

Both publish jobs use PyPI [trusted publishing](https://docs.pypi.org/trusted-publishers/) via the `pypi` / `testpypi` GitHub environments — no API tokens are stored in the repository.

## Resources
- [Meteo Swiss Open Data](https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html)
- [Jupyter Notebook Examples](https://github.com/MeteoSwiss/opendata-nwp-demos/tree/main)
- [Model Context Protocol](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Server Quickstart](https://modelcontextprotocol.info/docs/quickstart/server/)

## License
Licensed under the [Apache License 2.0](/LICENSE.txt).

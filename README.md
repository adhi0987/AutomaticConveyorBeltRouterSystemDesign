# AutomaticConveyorBeltRouterSystemDesign

## Project Overview

AutomaticConveyorBeltRouterSystemDesign is a prototype system that simulates and manages parcel routing on automated conveyor networks. The project includes a web client for visualization and control (`client/`) and a Python-based server for routing logic, simulation, and data processing (`server/`). It is intended for research, prototyping, and demonstration of routing algorithms, throughput optimization, and conveyor control strategies.

## Admin / Test Logins
- **Admin Test Login**: `username: admin`, `password: admin234`
- **Employee Test Login**: `username: test`, `password: test123`

**Note:** These credentials exist for testing only. Replace or remove them for any production deployment.

**Key Capabilities (Summary)**
- **Parcel routing simulation**: Route parcels to destinations using configurable algorithms and rules.
- **Visualization**: Web-based UI showing conveyors, parcel positions, and routing decisions in real time.
- **Batch data support**: Load and simulate using CSV datasets (e.g., `parcels_10000.csv`).
- **Extensible architecture**: Clear separation between `client` and `server` to add algorithms, sensors, and actuators.

## Project Structure

- `client/`: Frontend application (Vite + React + TypeScript)
	- `index.html`, `package.json`, `src/` — UI source files
- `server/`: Backend and simulation
	- `main.py` — server entrypoint (simulation and API)
	- `requirements.txt` — Python dependencies
	- `parcels_10000.csv` — sample dataset
	- `Dockerfile` — optional containerization

## Detailed Features & Functionalities

- **Routing Algorithms**: Implement and compare multiple routing strategies (e.g., shortest-path, priority-based, heuristic / ML-assisted routing).
- **Dynamic Re-routing**: Detect congestion and re-route parcels in-flight to avoid bottlenecks.
- **Prioritization & Scheduling**: Support for parcel priorities, deadlines, and SLA-aware routing.
- **Throughput Monitoring**: Real-time metrics (throughput, average delivery time, queue lengths).
- **Simulation Controls**: Start/pause/step simulation, adjust conveyor speed, injection rates, and fault injection.
- **Fault Tolerance & Recovery**: Simulate sensor/actuator failures and test recovery strategies.
- **Batch Import/Export**: Load parcel batches from CSV and export simulation traces for offline analysis.
- **API Endpoints**: REST or WebSocket endpoints for telemetry, control commands, and status updates.
- **Modular Plugin Interface**: Add new routing modules, sensors, or visualization components without changing core code.

## Suggested Roadmap (Additional Functionalities)

- **Machine Learning Integration**: Train models to predict congestion and optimize routing policies.
- **Digital Twin Integration**: Connect to real conveyor PLCs or digital twin frameworks for hardware-in-the-loop testing.
- **Authentication & RBAC**: Replace test credentials with real authentication and role-based access control.
- **User Workflows**: Add user roles, job management, and audit logs for operational use.
- **Advanced Analytics Dashboard**: Time-series visualization, alerts, and KPIs for operations teams.

## System Architecture

The system uses a client-server architecture with the following components:

- **Client (Frontend)**: A Vite + React application that presents the conveyor layout, live parcel positions, control widgets, and charts. It connects to the server via WebSocket or REST for low-latency updates.
- **Server (Backend / Simulator)**: A Python application that runs the simulation loop, executes routing algorithms, processes input CSV data, exposes APIs, and emits telemetry.
- **Data Layer**: CSV files used for batch simulations; optionally a lightweight DB (SQLite/Postgres) can store runs and metrics.
- **Transport Layer**: WebSocket for streaming simulation state and control commands; REST for administrative actions and history queries.
- **Optional Components**:
	- **Docker**: Containerize the server for consistent deployment.
	- **CI / Tests**: Unit and integration tests for routing logic.
	- **Monitoring**: Prometheus/Grafana for metrics and dashboards in production-style setups.

High-level flow:
1. Client sends start/simulation parameters to server.
2. Server ingests parcel batch (CSV) and initializes conveyor nodes.
3. The simulation loop advances parcel positions and invokes routing logic when junction decisions are required.
4. Server streams state updates to client for visualization.
5. Operators can change parameters (speed, priority rules) to observe effects in real time.

## Advantages

- **Cost-effective prototyping**: Enables testing routing policies without hardware.
- **Repeatable experiments**: Run identical datasets to compare algorithms and parameters.
- **Modular and extensible**: Clear separation of concerns makes it easy to add features.
- **Scalable simulation**: Use CSV batches (10k+ parcels) to stress test routing logic and measure throughput.
- **Improves operational decisions**: Simulate “what-if” scenarios before rolling out changes.

## Real-world Applications

- **E-commerce warehouses**: Optimize parcel sorting and routing to increase throughput and reduce transit times.
- **Distribution centers**: Simulate conveyor layouts and policies during facility planning.
- **Airport baggage handling**: Test routing and prioritization to reduce misroutes and delays.
- **Manufacturing lines**: Route parts between stations with priority and scheduling constraints.
- **Logistics testing labs**: Validate new sensors, PLC programs, or conveyor topologies safely.

## Installation & Usage

Prerequisites:
- Python 3.8+ for the server
- Node 16+ / npm for the client

Running the server (development):

1. Create and activate a Python virtual environment:

```powershell
python -m venv venv
; .\venv\Scripts\Activate.ps1
```

2. Install server dependencies:

```powershell
pip install -r server\requirements.txt
```

3. Run the server:

```powershell
python server\main.py
```

Running the client (development):

```powershell
cd client
npm install
npm run dev
```

Docker (optional):

1. Build image:

```powershell
docker build -t acbr-system -f server\Dockerfile .
```

2. Run container (example):

```powershell
docker run --rm -p 8000:8000 acbr-system
```

## Testing & Benchmarking

- Use `server/parcels_10000.csv` to run large-scale simulations and measure throughput.
- Add unit tests for routing modules and run via your preferred Python test runner.

## Contributing

- Fork the repo, open a branch, and submit PRs with clear descriptions.
- Add tests for new routing strategies or bug fixes.

## Contact & License

- **Author / Maintainer**: See repository owner `adhi0987`.
- This project is provided for educational and prototyping purposes. Replace test credentials before production use.

---

If you want, I can:
- run a quick local smoke test of the server and client (requires your environment), or
- add a simple architecture diagram (ASCII or SVG) inside this README.


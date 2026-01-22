I will refactor the project into a unified `RiskControl` application, strictly adhering to your requirements for style retention, core capability preservation, and functional integrity.

### 1. Unified Project Structure (Refactoring)
I will create a standardized Flask application structure under `RiskControl/app` to unify the disparate tools (`01-data_tool` and `04-risk_cot_tool`).

```text
RiskControl/
├── app/
│   ├── __init__.py          # Application Factory
│   ├── routes/              # Unified Routing Layer
│   │   ├── __init__.py
│   │   ├── main.py          # Dashboard Navigation (Home)
│   │   ├── data_tool.py     # Ported from 01-data_tool/run.py
│   │   └── risk_cot.py      # Ported from 04-risk_cot_tool/app/routes/
│   ├── services/            # Core Business Logic (Strictly Preserved)
│   │   ├── data_core/       # Moved from 01-data_tool/core/
│   │   └── risk_cot/        # Moved from 04-risk_cot_tool/app/services/
│   ├── templates/           # New UI Templates
│   │   ├── base.html        # Derived from stitch_.../code.html
│   │   ├── home.html
│   │   └── ...              # Feature-specific pages
│   └── static/              # Assets
├── data/                    # Unified Data Directory
├── logs/                    # Unified Logging
├── run.py                   # Single Entry Point
└── requirements.txt         # Unified Dependencies
```

### 2. Frontend Implementation (Style Retention)
*   **Base Template**: I will extract the Tailwind configuration, CSS variables, and layout structure from `stitch_cot_quality_inspection_detail/ai_pipeline_dashboard_overview_1/code.html` (Dark Mode) to create `app/templates/base.html`.
*   **Pixel-Perfect Migration**: The specific styles for "sidebar", "cards", and "nodes" will be copied verbatim to ensure the new UI matches the provided prototypes exactly.
*   **Navigation**: The sidebar will be updated to link to the migrated functionalities of both tools.

### 3. Backend Integration (Core Capability Retention)
*   **Data Tool Migration (`01-data_tool`)**:
    *   The `core` algorithms (PU Bagging, Feature Selection) will be moved to `app/services/data_core/`.
    *   The `run.py` logic will be refactored into `app/routes/data_tool.py`.
    *   **Crucial**: I will update the `subprocess` calls to dynamically locate the scripts in their new location, ensuring the logic remains identical (running as independent processes to maintain stability).
*   **Risk CoT Tool Migration (`04-risk_cot_tool`)**:
    *   The `app/services` directory will be moved to `app/services/risk_cot/` without modifying file contents.
    *   Existing Blueprints will be registered in the new application factory.

### 4. Verification & Testing
*   **UI Check**: Verify that the new Dashboard loads with the correct Dark Mode styling and Tailwind classes.
*   **Functional Check**:
    *   Verify `01-data_tool` endpoints (Upload, Run Model) work in the new structure.
    *   Verify `04-risk_cot_tool` endpoints (Inference, Inspection) are accessible.
*   **Impact Assessment**: I will confirm that no business logic code was altered, only file locations and import paths.

I will begin by creating the new directory structure and migrating the core assets.
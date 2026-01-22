# Core Functional Module Migration and Integration Plan

## 1. Functional Analysis & Architecture Design

### 1.1 Core Business Logic Analysis
- **Data Intelligence Core (`01-data_tool/core`)**:
  - **PU Learning**: Bagging-based positive-unlabeled learning for risk sample mining.
  - **Feature Selection**: Ensemble method combining XGBoost, Random Forest, and Mutual Information.
  - **Data Generation**: Simulation of financial risk datasets.
- **Risk COT Services (`04-risk_cot_tool/app/services`)**:
  - **Inference Engine**: Batch LLM inference (DeepSeek) with retry/resume capabilities.
  - **Model Inspector**: LLM-based quality scoring of Chain-of-Thought (COT).
  - **Rule Inspector**: Regex/Heuristic-based basic quality checks (truncation, formatting).
  - **Prompt Engine**: Automated prompt template generation and optimization.

### 1.2 Integration Architecture
Adopt a **Frontend-Backend Separation** pattern within the `stitch_cot_quality_inspection_detail` project structure:
- **Backend Layer**: Encapsulate the original Python scripts into a unified FastAPI service.
- **Frontend Layer**: Extend the existing static HTML dashboard with Alpine.js to interact with the backend.

## 2. Implementation Steps

### Phase 1: Backend Migration & Service Exposure
1.  **Directory Restructuring**:
    - Create `stitch_cot_quality_inspection_detail/backend/`.
    - Migrate `01-data_tool/core` -> `.../backend/core/`.
    - Migrate `04-risk_cot_tool/app/services` -> `.../backend/services/`.
2.  **API Development (`main.py`)**:
    - **Data APIs**:
        - `POST /api/data/pu-learning/train`: Trigger PU Bagging training.
        - `POST /api/data/feature-selection`: Run ensemble feature selection.
    - **COT APIs**:
        - `POST /api/cot/inference`: Start LLM inference tasks.
        - `POST /api/cot/inspect`: Run rule-based and model-based inspections.
3.  **Dependency Management**: Create `requirements.txt` aggregating dependencies from both source tools.

### Phase 2: Frontend Module Development
Create two new functional pages based on the existing `code.html` design system (Tailwind + Navy Blue Dark Mode):

1.  **Data Intelligence Dashboard (`data_engine.html`)**:
    - **PU Learning Panel**: File upload for Positive/Unlabeled data, configuration for `n_estimators`, and results display.
    - **Feature Engineering Panel**: Feature selection execution and Top-K feature visualization chart.
2.  **COT Factory Dashboard (`cot_factory.html`)**:
    - **Inference Studio**: Prompt template editor, Model config (API Key, Max Tokens), and Task Progress bar.
    - **Quality Inspector**: Interactive text input area to test "Rule Checks" (formatting) and "Model Checks" (logic scoring) in real-time.

### Phase 3: Technical Integration
- **UI Components**: Reuse the `sidebar-item`, `node-base`, and `config-card` styles to ensure visual consistency.
- **Logic Binding**: Use **Alpine.js** (`x-data`, `x-on:click`) to handle form submissions and API `fetch` calls.
- **Mock Data Support**: Ensure frontend handles loading states and displays mock results if the backend is not immediately reachable during preview.

## 3. Testing & Delivery
- **Unit Testing**: Create `tests/test_api.py` to verify the FastAPI endpoints invoke the underlying core logic correctly.
- **Integration Verification**: Verify that clicking "Run Inspection" on the frontend correctly returns JSON results from the backend.
- **Documentation**: Add a `README_INTEGRATION.md` explaining how to start the backend server and access the new pages.

## 4. Execution Order
1.  **Move Files**: Organize backend structure.
2.  **Create Backend**: Implement `main.py` and `requirements.txt`.
3.  **Create Frontend**: Implement `data_engine.html` and `cot_factory.html`.
4.  **Verify**: Run basic connectivity tests.

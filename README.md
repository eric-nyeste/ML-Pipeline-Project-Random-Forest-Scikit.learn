CSC 659/859 | AI Explainability and Ethics | Fall 2026
Eric Nyeste

An educational machine-learning experiment using Python and scikit-learn to classify single-cell gene-expression observations, evaluate model performance, and examine the tradeoffs between predictive accuracy and model complexity. The project also documents human oversight of generative-AI-assisted scientific programming.
Project overview
This study investigates Random Forest classification of single-cell gene-expression data, emphasizing model evaluation, feature selection, and methodological tradeoffs between predictive performance and model complexity. The original dataset contains 871 neuronal nuclei, 608 numerical gene-expression features, and a binary label indicating membership in the e1 excitatory neuron cluster (1) versus non-e1 (0).

Six Random Forest configurations were compared using out-of-bag (OOB) evaluation. The experiment then used manually constructed, stratified three-fold cross-validation, Gini-based feature-importance analysis, and reduced-feature classification. Feature selection was repeated within each cross-validation training fold to avoid leakage from the corresponding evaluation fold.
Research objectives
- Compare Random Forest performance across tree counts and feature-sampling settings.
- Contrast OOB evaluation with manually implemented stratified cross-validation.
- Examine whether smaller sets of gene-expression features preserve predictive performance.
- Evaluate classification errors, including false negatives for the e1 class.
- Document the role and limitations of human–AI collaboration in the research workflow.
Reported results
These are results reported in the accompanying research paper, not a claim that a fresh clone of this repository has independently reproduced them.
Experiment	Reported result
Best OOB configuration	100 trees, max_features="sqrt"
Best OOB accuracy	99.66%
Mean three-fold cross-validation accuracy	99.54%
Six-gene mean cross-validation accuracy	99.08%


The six-gene experiment achieved substantial dimensionality reduction but increased false-negative classifications relative to the full-feature approach. These results concern this dataset and experimental design; they do not establish generalization to independent donors or clinical use.
Dataset and provenance
The input dataset is a single-cell gene-expression CSV used in the course assignment and discussed in the accompanying report, which cites Aevermann et al. (2018) and the NS-Forest methodology. It contains 871 observations, with 299 e1 and 572 non-e1 labels. The samples originate from a single post-mortem brain, limiting population-level generalizability.

The pipeline expects a file named exactly:

Original training DB e1 cluster.csv

with a binary column named Label and 608 numeric feature columns. It withholds one observation from each class using random seed 42, creating 869 training observations and 2 runtime verification observations.

Data access: If the dataset is already committed to this repository, place or copy it into a local data/ folder under the exact filename above. If it is hosted elsewhere, download it from its documented source and place it there. Confirm the dataset's original source, citation, and redistribution license before redistributing or reusing it. The repository's uploaded CSV has not been independently checked against the source publication in preparing this README.
Repository files
File	Purpose
HW2_ML_Pipeline_Consolidated.py	Consolidated executable experiment: data split, OOB comparison, CV, feature selection, and runtime verification.
test_ml_pipeline.py	Unit and synthetic-data integration tests; optional original-data reproducibility check.
requirements.txt	Python package dependencies.
README.md	Project overview and reproducibility instructions.
Research report (.pdf)	Detailed methods, findings, interpretation, limitations, and AI-use disclosure.
Dataset (.csv, where permitted)	Original experimental observations.


The script and tests must be in the same directory. Dataset and report filenames may differ from the descriptions above; check the repository listing.
Installation
Requires a recent compatible Python 3 installation. The original experiment was documented as using Python 3.14.7, scikit-learn 1.9.0, and pandas 3.0.6. The dependency file pins the latter two; NumPy and Matplotlib are listed without exact version pins because the original versions were not documented.

From the repository's root directory:
python -m venv .venv
Activate the virtual environment:

Windows PowerShell
.\.venv\Scripts\Activate.ps1
macOS / Linux
source .venv/bin/activate
Then install dependencies:
python -m pip install -r requirements.txt
On Windows, if python is not recognized before activation, py -m venv .venv may work instead.
Running the experiment
Place the original CSV in data/, then run:
python HW2_ML_Pipeline_Consolidated.py --data-dir ./data --output-dir ./results
The script creates the output directory and writes evaluation tables and plots there. It also writes training_db.csv and verification_db.csv to the data directory, so keep a backup of any existing files with those names.

If you already have those two split files and want to use them without repartitioning the original data:
python HW2_ML_Pipeline_Consolidated.py --data-dir ./data --output-dir ./results --use-existing-split
Expected outputs include HW2_OOB_results.csv, HW2_CV_fold_results.csv, HW2_RF_Feature_Ranking.csv, HW2_Reduced_Feature_Pooled.csv, HW2_Runtime_Verification.csv, and HW2_RF_Top10_Features.png, along with additional summaries.
Running the tests
From the repository root:
python -m unittest -v test_ml_pipeline.py
The default suite tests dataset splitting, metric calculations, fold construction, missing-value checks, and a synthetic-data end-to-end run. Synthetic test data establish software behavior, not reproduction of the scientific findings.

To additionally check the reported OOB result against the original CSV, set HW2_DATA_DIR to the folder containing that CSV.

PowerShell:
$env:HW2_DATA_DIR = ".\data"
python -m unittest -v test_ml_pipeline.py
macOS / Linux:
HW2_DATA_DIR=./data python -m unittest -v test_ml_pipeline.py
The optional test checks the original dataset shape, class counts, and the reported three OOB errors in 869 training observations for the 100-tree, square-root model. It does not reproduce every result in the report. If HW2_DATA_DIR is unset, that test is skipped.
Reproducibility status
The consolidated script was assembled from the Python appendices of the course report with integration changes for portable paths, shared functions, command-line arguments, and noninteractive chart saving. It is not a byte-for-byte copy of the original separate scripts. A prior synthetic-data test run checked integration, but a full independent run against the original research CSV and confirmation of all published metrics remain necessary.

For a rigorous reproduction, run the pipeline in a clean environment, retain its output files, and compare OOB, cross-validation, reduced-feature, and runtime results with the report. Record package versions and any discrepancies.
Limitations and responsible interpretation
- Limited biological sampling: The dataset comes from one post-mortem brain and does not establish population-level generalizability.
- High dimensionality: 608 features relative to 871 original observations create a low sample-to-feature ratio.
- Reference labels: The classifier predicts labels derived from an existing biological classification procedure; these are not independently validated clinical outcomes.
- Performance uncertainty: High accuracy on this dataset does not guarantee reliability under distribution shift or on other donors.
- Reduced-feature tradeoffs: Fewer genes can preserve much of the reported accuracy while changing the error profile, including false negatives.
- Scope: This is an educational research experiment, not a validated diagnostic or clinical decision-making tool.
Generative-AI-assisted development
OpenAI ChatGPT was used as an instructional and computational collaborator for conceptual instruction, Python code generation, debugging, experimental-design consultation, statistical interpretation, and report review. The researcher retained responsibility for experimental choices, verification, interpretation, and the submitted conclusions. The research report provides additional detail on prompts, oversight, and the original scripts.
References
- Aevermann et al. (2018), source research discussed in the course report for the gene-expression data and NS-Forest context. Consult the PDF for the complete citation and original data provenance.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Full accompanying CSC 659/859 research report (PDF in this repository).
Keywords: Random Forest; supervised machine learning; gene-expression classification; feature selection; cross-validation; model evaluation; human–AI collaboration.

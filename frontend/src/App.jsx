import { useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_JOB_DESC =
  "Backend Developer with Java, Spring Boot, REST APIs, and SQL. Experience with Git and basic system design preferred.";

const API_URL = "http://127.0.0.1:5000/api/predict";
const METRICS_URL = "http://127.0.0.1:5000/api/metrics";

export default function App() {
  const [page, setPage] = useState("overview");
  const [jobDescription, setJobDescription] = useState(DEFAULT_JOB_DESC);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [metricsError, setMetricsError] = useState("");
  const fileInputRef = useRef(null);

  const fileCount = files.length;

  const canSubmit = useMemo(() => {
    return fileCount >= 1 && fileCount <= 100 && !loading;
  }, [fileCount, loading]);

  const handleFiles = (e) => {
    const selected = Array.from(e.target?.files || e.dataTransfer?.files || []);
    setFiles(selected);
    setResults([]);
    setError("");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFiles(e);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResults([]);

    if (files.length === 0) {
      setError("Please select at least 1 resume file.");
      return;
    }

    if (files.length > 100) {
      setError("Please select no more than 100 files.");
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("resumes", file));
    formData.append("job_description", jobDescription);

    try {
      setLoading(true);
      setProgress(8);
      const res = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Upload failed");
      }
      const sorted = (data.results || []).slice().sort((a, b) => {
        if (a.error && b.error) return 0;
        if (a.error) return 1;
        if (b.error) return -1;
        const ar = a.prediction === "Relevant" ? 1 : 0;
        const br = b.prediction === "Relevant" ? 1 : 0;
        if (br !== ar) return br - ar;
        const asem = a.semantic_score ?? -1;
        const bsem = b.semantic_score ?? -1;
        if (bsem !== asem) return bsem - asem;
        return (b.confidence || 0) - (a.confidence || 0);
      });
      setResults(sorted);
      setPage("insights");
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
      setProgress(100);
      setTimeout(() => setProgress(0), 800);
    }
  };

  useEffect(() => {
    if (!loading) return;
    let value = 12;
    const timer = setInterval(() => {
      value = Math.min(value + Math.random() * 8, 92);
      setProgress(Math.round(value));
    }, 350);
    return () => clearInterval(timer);
  }, [loading]);

  useEffect(() => {
    if (page !== "insights") return;
    let active = true;
    setMetricsError("");
    fetch(METRICS_URL)
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        if (data?.error) {
          setMetricsError(data.error);
          setMetrics(null);
          return;
        }
        setMetrics(data);
      })
      .catch(() => {
        if (!active) return;
        setMetricsError("Metrics unavailable");
        setMetrics(null);
      });
    return () => {
      active = false;
    };
  }, [page]);

  const stats = useMemo(() => {
    const ok = results.filter((r) => !r.error);
    const relevant = ok.filter((r) => r.prediction === "Relevant").length;
    const notRelevant = ok.filter((r) => r.prediction === "Not Relevant").length;
    const avgConfidence =
      ok.length === 0
        ? 0
        : ok.reduce((sum, r) => sum + (r.confidence || 0), 0) / ok.length;
    const avgSemantic =
      ok.length === 0
        ? 0
        : ok.reduce((sum, r) => sum + (r.semantic_score || 0), 0) / ok.length;
    const byCategory = ok.reduce((acc, r) => {
      const key = r.predicted_category || "Unknown";
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
    const semanticScores = ok
      .map((r) => (typeof r.semantic_score === "number" ? r.semantic_score : null))
      .filter((v) => v !== null);
    const bins = [0, 0.25, 0.5, 0.75, 1.0];
    const semanticBins = bins.slice(0, -1).map((start, i) => {
      const end = bins[i + 1];
      const count = semanticScores.filter(
        (v) => v >= start && (i === bins.length - 2 ? v <= end : v < end)
      ).length;
      return { label: `${start.toFixed(2)}–${end.toFixed(2)}`, count };
    });
    const topCategories = Object.entries(byCategory)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6);
    return {
      total: results.length,
      relevant,
      notRelevant,
      avgConfidence,
      avgSemantic,
      semanticBins,
      topCategories,
    };
  }, [results]);

  const macro = metrics?.report?.["macro avg"] || null;

  return (
    <div className="page">
      <nav className="topbar">
        <div className="brandmark">
          <span className="logo">TL</span>
          TalentLens
        </div>
        <div className="nav">
          {[
            ["overview", "Overview"],
            ["screen", "Screen"],
            ["insights", "Insights"],
            ["about", "About"],
          ].map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={`nav-link ${page === key ? "active" : ""}`}
              onClick={() => setPage(key)}
            >
              {label}
            </button>
          ))}
        </div>
        <button className="cta" type="button" onClick={() => setPage("screen")}>
          Start Screening
        </button>
      </nav>

      {page === "overview" && (
        <section className="hero">
          <div className="hero-grid">
            <div>
              <div className="eyebrow">Explainable ML · Resume Intelligence</div>
              <h1>Professional Resume Screening for Hiring Teams</h1>
              <p>
                Upload a batch of resumes, define your role, and get relevance decisions
                powered by a transparent ML pipeline. Built for speed, clarity, and trust.
              </p>
              <div className="hero-actions">
                <button className="button" type="button" onClick={() => setPage("screen")}>
                  Screen Resumes
                </button>
                <button className="button ghost" type="button" onClick={() => setPage("insights")}>
                  View Insights
                </button>
              </div>
            </div>
            <div className="hero-card">
              <div className="metric">
                <span>Average Confidence</span>
                <strong>{stats.avgConfidence.toFixed(2)}</strong>
              </div>
              <div className="metric">
                <span>Average Semantic</span>
                <strong>{stats.avgSemantic.toFixed(2)}</strong>
              </div>
              <div className="metric">
                <span>Files Screened</span>
                <strong>{stats.total}</strong>
              </div>
              <div className="tagline">Category-based screening with interpretable scores.</div>
            </div>
          </div>
        </section>
      )}

      {page === "screen" && (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Screen Resumes</h2>
              <p className="muted">
                The system infers a target category from your job description and
                matches resumes based on predicted category.
              </p>
            </div>
            <div className="chip">Max 100 files</div>
          </div>
          <form onSubmit={handleSubmit} className="form">
            <label className="label">Job Description</label>
            <textarea
              className="textarea"
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste job description"
            />

            <label className="label">Upload Resumes (TXT, PDF, DOCX)</label>
            <div
              className={`upload ${isDragOver ? "drag" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragOver(true);
              }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.pdf,.docx"
                onChange={handleFiles}
                className="file-input"
              />
              <div className="upload-row">
                <div>
                  <div className="upload-title">Drag & drop your resumes</div>
                  <div className="meta">or select files from your device</div>
                </div>
                <button
                  type="button"
                  className="button ghost"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Choose Files
                </button>
              </div>
              <div className="meta">
                {fileCount === 0 && "No files selected"}
                {fileCount === 1 && "1 file selected"}
                {fileCount > 1 && `${fileCount} files selected`}
              </div>
            </div>

            <button className="button" type="submit" disabled={!canSubmit}>
              {loading ? "Screening..." : "Screen Resumes"}
            </button>

            {loading && (
              <div className="progress-wrap">
                <div className="progress-bar">
                  <div className="progress" style={{ width: `${progress}%` }} />
                </div>
                <div className="progress-text">{progress}% completed</div>
              </div>
            )}

            {error && <div className="error">{error}</div>}
          </form>
        </section>
      )}

      {page === "insights" && (
        <section className="results">
          <div className="results-header">
            <div>
              <h2>Screening Insights</h2>
              <p className="muted">Performance and distribution from your latest run.</p>
            </div>
            <span className="chip">{results.length} processed</span>
          </div>

          {results.length === 0 && (
            <div className="empty">Upload resumes to see results here.</div>
          )}

          {results.length > 0 && (
            <>
              <div className="kpi-grid">
                <div className="kpi">
                  <span>Relevant</span>
                  <strong>{stats.relevant}</strong>
                </div>
                <div className="kpi">
                  <span>Not Relevant</span>
                  <strong>{stats.notRelevant}</strong>
                </div>
                <div className="kpi">
                  <span>Avg Confidence</span>
                  <strong>{stats.avgConfidence.toFixed(2)}</strong>
                </div>
                <div className="kpi">
                  <span>Avg Semantic</span>
                  <strong>{stats.avgSemantic.toFixed(2)}</strong>
                </div>
              </div>

              <div className="metrics-grid">
                <div className="metric-card">
                  <h3>Model Evaluation</h3>
                  {metricsError && <div className="error">{metricsError}</div>}
                  {!metricsError && metrics && (
                    <div className="metric-list">
                      <div>
                        <span>Accuracy</span>
                        <strong>{metrics.accuracy.toFixed(2)}</strong>
                      </div>
                      <div>
                        <span>Macro Precision</span>
                        <strong>{macro?.precision?.toFixed(2) || "-"}</strong>
                      </div>
                      <div>
                        <span>Macro Recall</span>
                        <strong>{macro?.recall?.toFixed(2) || "-"}</strong>
                      </div>
                      <div>
                        <span>Macro F1</span>
                        <strong>{macro?.["f1-score"]?.toFixed(2) || "-"}</strong>
                      </div>
                    </div>
                  )}
                </div>
                <div className="metric-card">
                  <h3>Ranking Logic</h3>
                  <p className="muted">
                    Resumes are ranked by relevance first, then semantic similarity,
                    then confidence.
                  </p>
                </div>
              </div>

              <div className="charts">
                <div className="chart-card">
                  <h3>Relevance Split</h3>
                  <div className="donut">
                    <div
                      className="slice"
                      style={{
                        "--p": stats.total ? stats.relevant / stats.total : 0,
                      }}
                    />
                    <div className="donut-center">
                      {stats.total ? Math.round((stats.relevant / stats.total) * 100) : 0}%
                    </div>
                  </div>
                  <div className="legend">
                    <span className="dot green" /> Relevant
                    <span className="dot amber" /> Not Relevant
                  </div>
                </div>
                <div className="chart-card">
                  <h3>Top Predicted Categories</h3>
                  <div className="bars">
                    {stats.topCategories.map(([label, value]) => (
                      <div key={label} className="bar-row">
                        <span>{label}</span>
                        <div className="bar">
                          <div
                            className="bar-fill"
                            style={{
                              width: `${(value / Math.max(1, stats.total)) * 100}%`,
                            }}
                          />
                        </div>
                        <strong>{value}</strong>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="chart-card">
                  <h3>Semantic Score Distribution</h3>
                  <div className="bars">
                    {stats.semanticBins.map((bin) => (
                      <div key={bin.label} className="bar-row">
                        <span>{bin.label}</span>
                        <div className="bar">
                          <div
                            className="bar-fill"
                            style={{
                              width: `${stats.total ? (bin.count / stats.total) * 100 : 0}%`,
                            }}
                          />
                        </div>
                        <strong>{bin.count}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}

          <div className="cards">
            {results.map((r, idx) => (
              <div
                key={r.filename}
                className={`card ${r.prediction === "Not Relevant" ? "not-relevant" : ""}`}
              >
                <div className="file">#{idx + 1} · {r.filename}</div>
                {r.error ? (
                  <div className="error">{r.error}</div>
                ) : (
                  <>
                    <div className={`tag ${r.prediction === "Not Relevant" ? "not-relevant" : ""}`}>
                      {r.prediction}
                    </div>
                    <div className="score">
                      Predicted: {r.predicted_category}
                    </div>
                    <div className="score">
                      Confidence: {r.confidence.toFixed(2)}
                    </div>
                    {r.semantic_score !== null && r.semantic_score !== undefined && (
                      <div className="score">Semantic: {r.semantic_score.toFixed(2)}</div>
                    )}
                    {Array.isArray(r.matched_skills) && r.matched_skills.length > 0 && (
                      <div className="skills">
                        {r.matched_skills.map((skill) => (
                          <span key={skill} className="skill-chip">
                            {skill}
                          </span>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {page === "about" && (
        <section className="about">
          <div className="about-card">
            <h2>How It Works</h2>
            <p>
              The system vectorizes resume text using TF-IDF, then applies a trained
              Logistic Regression classifier to predict the most likely resume category.
              The job description is analyzed to infer a target category, and resumes
              are marked Relevant if they match that target.
            </p>
            <div className="pill-grid">
              <div className="pill">Explainable ML</div>
              <div className="pill">Batch Upload</div>
              <div className="pill">Fast Scoring</div>
              <div className="pill">Category Mapping</div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

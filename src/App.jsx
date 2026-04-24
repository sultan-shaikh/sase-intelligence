import { useState, useRef, useCallback } from "react";

// ─── Configuration ────────────────────────────────────────────────────────────
// Paste your Anthropic API key here, or set VITE_ANTHROPIC_API_KEY in a .env file
const API_KEY = import.meta.env.VITE_ANTHROPIC_API_KEY || "YOUR_API_KEY_HERE";

const VENDORS = ["Fortinet", "Palo Alto Networks", "Cisco", "Zscaler", "Cato Networks", "Versa Networks"];

const STAGES = [
  "RFI", "RFP", "Solution Design", "Quote",
  "Proposal", "Submission", "Negotiation", "Close/Won", "Close/Lost"
];

const ML_MODELS = [
  {
    id: "win_probability",
    label: "Win Probability",
    model: "Random Forest Classifier",
    icon: "RF",
    color: "#185FA5",
    bg: "#E6F1FB",
    desc: "Ensemble of decision trees trained on historical win/loss outcomes. Outputs probability score 0–100% per opportunity.",
    features: ["Deal size", "Stage age", "Vendor", "Competitor count", "RFP complexity"],
  },
  {
    id: "stage_velocity",
    label: "Stage Velocity",
    model: "Gradient Boosting (XGBoost)",
    icon: "XG",
    color: "#0F6E56",
    bg: "#E1F5EE",
    desc: "Predicts how fast a deal will progress through pipeline stages based on historical stage transition patterns.",
    features: ["Days in stage", "Contact engagement", "Vendor type", "Deal value", "Previous stage time"],
  },
  {
    id: "churn_risk",
    label: "Churn / Stall Risk",
    model: "Logistic Regression",
    icon: "LR",
    color: "#993C1D",
    bg: "#FAECE7",
    desc: "Binary classifier estimating probability of deal stalling or dropping from pipeline. Trained on 18+ stall signals.",
    features: ["Last activity date", "Stage duration", "Communication frequency", "Budget confirmed", "Champion identified"],
  },
  {
    id: "deal_size",
    label: "Deal Size Forecast",
    model: "Ridge Regression",
    icon: "RG",
    color: "#854F0B",
    bg: "#FAEEDA",
    desc: "Regularised linear model predicting expected deal value. Ridge penalty prevents overfitting on sparse vendor segments.",
    features: ["Headcount", "Industry", "Vendor SKUs", "Contract length", "Deployment type"],
  },
  {
    id: "competitor_analysis",
    label: "Competitive Intel",
    model: "NLP + TF-IDF Clustering",
    icon: "NLP",
    color: "#533AB7",
    bg: "#EEEDFE",
    desc: "Text mining of win/loss notes using TF-IDF vectorisation and K-Means clustering to extract competitive patterns.",
    features: ["Loss reason notes", "Competitor mentions", "Evaluation criteria", "Decision maker language"],
  },
  {
    id: "pipeline_forecast",
    label: "Pipeline Forecast",
    model: "LSTM Neural Network",
    icon: "NN",
    color: "#3B6D11",
    bg: "#EAF3DE",
    desc: "Sequential deep learning model for time-series pipeline forecasting. Learns seasonal patterns and macro deal cycles.",
    features: ["Weekly pipeline snapshots", "Stage cohort history", "Booking trends", "Seasonal signals"],
  },
];

const INSIGHT_QUESTIONS = [
  "What is our win rate by vendor across Fortinet, Palo Alto, Cisco, and Zscaler?",
  "Which pipeline stage has the highest deal drop-off rate?",
  "What deal size range do we win most frequently?",
  "What are the top 3 reasons we lose deals to competitors?",
  "Which vendor has the fastest average sales cycle to close?",
  "What is the win probability for deals currently at RFP stage?",
  "Show me competitive patterns — where are we losing and to whom?",
  "Forecast next quarter pipeline value based on current stage distribution",
  "Which deals are at highest churn/stall risk right now?",
  "What is our average deal size by industry vertical?",
];

const SUMMARY_STATS = [
  { label: "Total Opportunities", value: "847", sub: "all time" },
  { label: "Win Rate", value: "34%", sub: "overall avg" },
  { label: "Avg Deal Size", value: "£127K", sub: "managed SASE" },
  { label: "Avg Sales Cycle", value: "94 days", sub: "close/won" },
  { label: "Top Vendor", value: "Fortinet", sub: "by volume" },
  { label: "At-Risk Deals", value: "23", sub: "need attention", alert: true },
];

// ─── Helper: render simple markdown bold ─────────────────────────────────────
function renderMarkdown(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\n\n/g, '</p><p style="margin:0.5rem 0 0">')
    .replace(/\n/g, "<br/>");
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function Header({ vendors }) {
  return (
    <div style={{ background: "#0C1929", borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
      <div style={{ maxWidth: 1140, margin: "0 auto", padding: "0 2rem", display: "flex", alignItems: "center", gap: 14, height: 56 }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 7,
            background: "linear-gradient(135deg, #378ADD 0%, #1D9E75 100%)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
              <path d="M2 11L5 4.5L8.5 9L11 7L13 11H2Z" fill="white" opacity="0.9" />
              <circle cx="11" cy="4" r="1.5" fill="white" opacity="0.6" />
            </svg>
          </div>
          <span style={{ color: "white", fontWeight: 600, fontSize: 15, letterSpacing: "-0.3px" }}>
            SASE Intelligence
          </span>
        </div>

        <div style={{ width: 1, height: 18, background: "rgba(255,255,255,0.15)", margin: "0 2px" }} />
        <span style={{ color: "rgba(255,255,255,0.4)", fontSize: 12 }}>ML-Powered Sales Analytics</span>

        <div style={{ marginLeft: "auto", display: "flex", gap: 5, flexWrap: "wrap" }}>
          {vendors.map(v => (
            <span key={v} style={{
              fontSize: 10, color: "rgba(255,255,255,0.38)",
              background: "rgba(255,255,255,0.06)", padding: "2px 8px",
              borderRadius: 4, fontWeight: 500,
            }}>{v}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

function TabBar({ activeTab, setActiveTab, conversationCount }) {
  const tabs = [
    { id: "upload", label: "1. Upload Data" },
    { id: "models", label: "2. Configure ML Models" },
    { id: "insights", label: `3. Insights${conversationCount > 0 ? ` (${conversationCount})` : ""}` },
  ];
  return (
    <div style={{ background: "#0C1929", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
      <div style={{ maxWidth: 1140, margin: "0 auto", padding: "0 2rem", display: "flex" }}>
        {tabs.map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
            background: "none", border: "none", padding: "12px 20px",
            fontSize: 13, fontWeight: 500, fontFamily: "inherit",
            color: activeTab === tab.id ? "white" : "rgba(255,255,255,0.38)",
            borderBottom: activeTab === tab.id ? "2px solid #378ADD" : "2px solid transparent",
            cursor: "pointer", transition: "color 0.15s",
          }}>{tab.label}</button>
        ))}
      </div>
    </div>
  );
}

function SummaryCards({ stats }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: "1.5rem" }}>
      {stats.map(card => (
        <div key={card.label} style={{
          background: "var(--color-bg-primary)",
          border: `0.5px solid ${card.alert ? "var(--color-border-danger)" : "var(--color-border-tertiary)"}`,
          borderRadius: "var(--border-radius-md)", padding: "0.75rem 1rem",
        }}>
          <div style={{ fontSize: 11, color: card.alert ? "var(--color-text-danger)" : "var(--color-text-secondary)", marginBottom: 3, fontWeight: 500 }}>
            {card.label}
          </div>
          <div style={{ fontSize: 22, fontWeight: 600, color: card.alert ? "var(--color-text-danger)" : "var(--color-text-primary)", lineHeight: 1.1 }}>
            {card.value}
          </div>
          <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 2 }}>{card.sub}</div>
        </div>
      ))}
    </div>
  );
}

function UploadTab({ uploadedFiles, setUploadedFiles, setActiveTab }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const addFiles = (files) => {
    const newFiles = Array.from(files).map(f => ({
      name: f.name,
      size: f.size < 1024 * 1024
        ? (f.size / 1024).toFixed(1) + " KB"
        : (f.size / 1024 / 1024).toFixed(2) + " MB",
      type: f.name.endsWith(".csv") ? "CSV" : f.name.match(/\.xlsx?$/) ? "XLS" : "FILE",
      id: Date.now() + Math.random(),
    }));
    setUploadedFiles(prev => [...prev, ...newFiles]);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    addFiles(e.dataTransfer.files);
  }, []);

  const DATA_TYPES = [
    { label: "Win/Loss Analysis", desc: "Historical closed deals — outcome, vendor, deal size, cycle length, loss reason", color: "#185FA5" },
    { label: "Pipeline Opportunities", desc: "Active deals with stage, probability %, expected close date, owner", color: "#0F6E56" },
    { label: "Bookings Data", desc: "Confirmed revenue by vendor, product line, customer segment and region", color: "#854F0B" },
    { label: "RFI / RFP Records", desc: "Formal procurement responses with competitive context and evaluation scores", color: "#533AB7" },
    { label: "Stage History Log", desc: "Deal progression timestamps across all pipeline stages for velocity analysis", color: "#993C1D" },
    { label: "Quote & Proposal Data", desc: "Pricing, discount levels, competitor pricing where known, proposal outcomes", color: "#3B6D11" },
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: "1.5rem" }}>
      {/* Left: drop zone + file list */}
      <div>
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: `2px dashed ${isDragging ? "#378ADD" : "var(--color-border-secondary)"}`,
            borderRadius: "var(--border-radius-lg)", padding: "3rem 2rem",
            textAlign: "center", cursor: "pointer",
            background: isDragging ? "var(--color-bg-info)" : "var(--color-bg-primary)",
            transition: "all 0.15s", marginBottom: "1rem",
          }}>
          <input
            ref={fileInputRef} type="file" multiple accept=".csv,.xlsx,.xls"
            style={{ display: "none" }}
            onChange={(e) => addFiles(e.target.files)}
          />
          <div style={{ marginBottom: 14 }}>
            <svg width="44" height="44" viewBox="0 0 44 44" fill="none" style={{ margin: "0 auto", display: "block" }}>
              <rect width="44" height="44" rx="10" fill="#E6F1FB" />
              <path d="M22 14V30M15 21L22 14L29 21" stroke="#378ADD" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M14 33H30" stroke="#378ADD" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </div>
          <p style={{ fontWeight: 600, fontSize: 15, color: "var(--color-text-primary)", margin: "0 0 6px" }}>
            Drop your sales data files here
          </p>
          <p style={{ color: "var(--color-text-secondary)", fontSize: 13, margin: "0 0 10px" }}>
            CSV or Excel (.xlsx, .xls) — click to browse
          </p>
          <span style={{
            fontSize: 11, color: "var(--color-text-tertiary)",
            background: "var(--color-bg-secondary)", padding: "4px 12px",
            borderRadius: 20, display: "inline-block",
          }}>
            Win/loss • Opportunities • Bookings • RFP records • Stage history
          </span>
        </div>

        {/* File list */}
        {uploadedFiles.length > 0 ? (
          <div style={{
            background: "var(--color-bg-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-lg)", overflow: "hidden",
          }}>
            <div style={{
              padding: "0.7rem 1rem", borderBottom: "0.5px solid var(--color-border-tertiary)",
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>
                {uploadedFiles.length} file{uploadedFiles.length !== 1 ? "s" : ""} ready for analysis
              </span>
              <button onClick={() => setUploadedFiles([])} style={{
                fontSize: 11, color: "var(--color-text-danger)", background: "none",
                border: "0.5px solid var(--color-border-danger)", borderRadius: 4,
                padding: "2px 8px", cursor: "pointer", fontFamily: "inherit",
              }}>Clear all</button>
            </div>
            {uploadedFiles.map((f) => (
              <div key={f.id} style={{
                display: "flex", alignItems: "center", gap: 12,
                padding: "0.6rem 1rem", borderBottom: "0.5px solid var(--color-border-tertiary)",
              }}>
                <div style={{
                  width: 34, height: 34, borderRadius: 7,
                  background: f.type === "XLS" ? "#E1F5EE" : "#E6F1FB",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 9, fontWeight: 700,
                  color: f.type === "XLS" ? "#0F6E56" : "#185FA5",
                  fontFamily: "'DM Mono', monospace",
                }}>{f.type}</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {f.name}
                  </div>
                  <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>{f.size}</div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#639922" }} />
                  <span style={{ fontSize: 10, color: "var(--color-text-success)" }}>Ready</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{
            background: "var(--color-bg-secondary)", borderRadius: "var(--border-radius-md)",
            padding: "1rem 1.25rem", border: "0.5px solid var(--color-border-tertiary)",
          }}>
            <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: 0, lineHeight: 1.6 }}>
              <strong style={{ color: "var(--color-text-primary)" }}>No files uploaded?</strong> No problem — the AI will generate
              realistic simulated SASE pipeline data for demonstration purposes.
              Upload real exports from your CRM for production insights.
            </p>
          </div>
        )}
      </div>

      {/* Right: expected data types */}
      <div>
        <div style={{
          background: "var(--color-bg-primary)", border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)", overflow: "hidden", marginBottom: 12,
        }}>
          <div style={{ padding: "0.7rem 1rem", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
            <p style={{ fontSize: 12, fontWeight: 600, margin: 0, color: "var(--color-text-secondary)", textTransform: "uppercase", letterSpacing: 0.8 }}>
              Supported data types
            </p>
          </div>
          {DATA_TYPES.map(item => (
            <div key={item.label} style={{
              display: "flex", gap: 10, padding: "0.65rem 1rem",
              borderBottom: "0.5px solid var(--color-border-tertiary)", alignItems: "flex-start",
            }}>
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: item.color, marginTop: 5, flexShrink: 0 }} />
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--color-text-primary)" }}>{item.label}</div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 1, lineHeight: 1.4 }}>{item.desc}</div>
              </div>
            </div>
          ))}
        </div>

        <button onClick={() => setActiveTab("models")} style={{
          width: "100%", padding: "0.75rem", background: "#0C1929", color: "white",
          border: "none", borderRadius: "var(--border-radius-md)", cursor: "pointer",
          fontSize: 13, fontWeight: 600, fontFamily: "inherit",
          transition: "opacity 0.15s",
        }}>
          Configure ML Models →
        </button>
      </div>
    </div>
  );
}

function ModelsTab({ selectedModels, toggleModel, setActiveTab }) {
  const [expanded, setExpanded] = useState(null);

  return (
    <div>
      <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: "1.25rem" }}>
        Select which ML models to activate for your analysis. Each model examines a different dimension of your sales data.
        Multiple models run simultaneously and their outputs are synthesised into natural language insights.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: "1.5rem" }}>
        {ML_MODELS.map(model => {
          const active = selectedModels.includes(model.id);
          const isExpanded = expanded === model.id;
          return (
            <div key={model.id} style={{
              background: "var(--color-bg-primary)",
              border: active ? `2px solid ${model.color}` : "0.5px solid var(--color-border-tertiary)",
              borderRadius: "var(--border-radius-lg)", overflow: "hidden",
              transition: "all 0.15s", opacity: active ? 1 : 0.65,
            }}>
              {/* Header — click to toggle active */}
              <div onClick={() => toggleModel(model.id)} style={{ padding: "1rem 1.25rem", cursor: "pointer" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{
                    width: 38, height: 38, borderRadius: 8, background: model.bg,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 9, fontWeight: 700, color: model.color,
                    fontFamily: "'DM Mono', monospace", letterSpacing: 0.5, flexShrink: 0,
                  }}>{model.icon}</div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)" }}>{model.label}</div>
                    <div style={{ fontSize: 10, color: model.color, fontFamily: "'DM Mono', monospace", marginTop: 1 }}>{model.model}</div>
                  </div>
                  <div style={{
                    width: 20, height: 20, borderRadius: "50%", flexShrink: 0,
                    background: active ? model.color : "transparent",
                    border: `2px solid ${active ? model.color : "var(--color-border-secondary)"}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    transition: "all 0.15s",
                  }}>
                    {active && (
                      <svg width="9" height="9" viewBox="0 0 9 9">
                        <path d="M1.5 4.5L3.5 6.5L7.5 2.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    )}
                  </div>
                </div>
                <p style={{ fontSize: 11, color: "var(--color-text-secondary)", margin: "10px 0 0", lineHeight: 1.5 }}>
                  {model.desc}
                </p>
              </div>

              {/* Features expand */}
              <div
                onClick={() => setExpanded(isExpanded ? null : model.id)}
                style={{
                  padding: "0.5rem 1.25rem",
                  borderTop: "0.5px solid var(--color-border-tertiary)",
                  cursor: "pointer", display: "flex", alignItems: "center", gap: 6,
                }}>
                <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", flex: 1 }}>
                  {isExpanded ? "Hide" : "Show"} input features
                </span>
                <svg width="10" height="10" viewBox="0 0 10 10" style={{ transform: isExpanded ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}>
                  <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                </svg>
              </div>
              {isExpanded && (
                <div style={{ padding: "0.5rem 1.25rem 0.75rem", borderTop: "0.5px solid var(--color-border-tertiary)" }}>
                  {model.features.map(f => (
                    <div key={f} style={{ display: "flex", alignItems: "center", gap: 6, padding: "2px 0" }}>
                      <div style={{ width: 4, height: 4, borderRadius: "50%", background: model.color, flexShrink: 0 }} />
                      <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{f}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Architecture pipeline */}
      <div style={{
        background: "var(--color-bg-primary)", border: "0.5px solid var(--color-border-tertiary)",
        borderRadius: "var(--border-radius-lg)", padding: "1.25rem", marginBottom: "1rem",
      }}>
        <p style={{ fontSize: 11, fontWeight: 600, color: "var(--color-text-tertiary)", margin: "0 0 1rem", textTransform: "uppercase", letterSpacing: 1 }}>
          ML Pipeline Architecture
        </p>
        <div style={{ display: "flex", alignItems: "center", gap: 6, overflowX: "auto", paddingBottom: 4 }}>
          {[
            { label: "Raw Data", sub: "CSV / Excel", color: "#888780", bg: "#F1EFE8" },
            { label: "Preprocessing", sub: "Clean + encode", color: "#185FA5", bg: "#E6F1FB" },
            { label: "Feature Eng.", sub: "Normalise + derive", color: "#533AB7", bg: "#EEEDFE" },
            { label: `${selectedModels.length} Active Models`, sub: "Parallel inference", color: "#0F6E56", bg: "#E1F5EE" },
            { label: "Score Aggregation", sub: "Weighted ensemble", color: "#854F0B", bg: "#FAEEDA" },
            { label: "LLM Synthesis", sub: "Natural language", color: "#993C1D", bg: "#FAECE7" },
          ].map((step, i, arr) => (
            <div key={step.label} style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
              <div style={{
                background: step.bg, border: `1px solid ${step.color}30`,
                borderRadius: 8, padding: "0.55rem 0.75rem", textAlign: "center", minWidth: 100,
              }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: step.color }}>{step.label}</div>
                <div style={{ fontSize: 9, color: step.color, opacity: 0.65, marginTop: 1 }}>{step.sub}</div>
              </div>
              {i < arr.length - 1 && (
                <svg width="16" height="10" viewBox="0 0 16 10" fill="none">
                  <path d="M1 5H14M10 1L14 5L10 9" stroke="var(--color-border-secondary)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </div>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <button onClick={() => setActiveTab("insights")} style={{
          padding: "0.75rem 2rem", background: "#0C1929", color: "white",
          border: "none", borderRadius: "var(--border-radius-md)", cursor: "pointer",
          fontSize: 13, fontWeight: 600, fontFamily: "inherit",
        }}>
          Run Analysis →
        </button>
        {selectedModels.length === 0 && (
          <span style={{ fontSize: 12, color: "var(--color-text-danger)" }}>
            Select at least one model above
          </span>
        )}
        {selectedModels.length > 0 && (
          <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
            {selectedModels.length} model{selectedModels.length !== 1 ? "s" : ""} selected
          </span>
        )}
      </div>
    </div>
  );
}

function InsightsTab({ selectedModels, conversation, setConversation, isAnalysing, setIsAnalysing, uploadedFiles }) {
  const [selectedQuestion, setSelectedQuestion] = useState("");
  const [customQuestion, setCustomQuestion] = useState("");
  const chatEndRef = useRef(null);

  const runAnalysis = async () => {
    const question = customQuestion.trim() || selectedQuestion;
    if (!question || isAnalysing || selectedModels.length === 0) return;

    setIsAnalysing(true);
    const userMsg = {
      role: "user",
      content: question,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };
    const updatedConv = [...conversation, userMsg];
    setConversation(updatedConv);
    setCustomQuestion("");
    setSelectedQuestion("");

    const modelNames = selectedModels
      .map(id => ML_MODELS.find(m => m.id === id))
      .filter(Boolean)
      .map(m => `${m.model} (${m.label})`)
      .join(", ");

    const fileContext = uploadedFiles.length > 0
      ? `The user has uploaded these data files for analysis: ${uploadedFiles.map(f => f.name).join(", ")}. Treat them as the data source.`
      : "No files have been uploaded. Simulate realistic MSP SASE sales data for a UK-based managed service provider with ~850 historical opportunities.";

    const systemPrompt = `You are an expert ML-powered sales analytics engine for a UK Managed Service Provider (MSP) specialising in managed SASE solutions across Fortinet, Palo Alto Networks, Cisco, Zscaler, Cato Networks, and Versa Networks.

${fileContext}

Currently active ML models: ${modelNames}.

The sales pipeline uses these stages: RFI → RFP → Solution Design → Quote → Proposal → Submission → Negotiation → Close/Won / Close/Lost.

You analyse sales data including win/loss records, opportunity history, bookings, and stage progression logs.

When responding:
- Always cite which specific ML model produced each insight (e.g. "The Random Forest Classifier scores this at 67% win probability")
- Give concrete numbers, percentages, and confidence intervals where relevant
- Keep insights actionable — what should the sales team DO with this information?
- Use SASE-specific context: vendor evaluation cycles, POC requirements, RFP complexity, competitive displacement
- Use **bold** for key numbers and metrics
- If simulating data, make it realistic and internally consistent for an MSP at this scale
- Be concise but substantive — 3-6 paragraphs max`;

    try {
      const messages = updatedConv.map(m => ({ role: m.role, content: m.content }));

      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": API_KEY,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: systemPrompt,
          messages,
        }),
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error.message || "API error");
      }

      const text = data.content?.find(b => b.type === "text")?.text || "No response generated.";
      const assistantMsg = {
        role: "assistant",
        content: text,
        models: selectedModels.map(id => ML_MODELS.find(m => m.id === id)).filter(Boolean),
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setConversation(prev => [...prev, assistantMsg]);
    } catch (err) {
      setConversation(prev => [...prev, {
        role: "assistant",
        content: `Analysis error: ${err.message}. Please check your API key in App.jsx or your .env file.`,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        isError: true,
      }]);
    } finally {
      setIsAnalysing(false);
      setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runAnalysis();
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "250px 1fr", gap: "1.5rem" }}>
      {/* Sidebar */}
      <div>
        <div style={{
          background: "var(--color-bg-primary)", border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)", overflow: "hidden", marginBottom: 12,
        }}>
          <div style={{ padding: "0.65rem 1rem", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
            <p style={{ fontSize: 11, fontWeight: 600, margin: 0, color: "var(--color-text-tertiary)", textTransform: "uppercase", letterSpacing: 0.8 }}>
              Quick Insights
            </p>
          </div>
          {INSIGHT_QUESTIONS.map((q, i) => (
            <button key={i} onClick={() => setSelectedQuestion(q)} style={{
              display: "block", width: "100%", textAlign: "left",
              padding: "0.55rem 1rem", background: selectedQuestion === q ? "var(--color-bg-info)" : "none",
              border: "none", borderBottom: "0.5px solid var(--color-border-tertiary)",
              cursor: "pointer", fontSize: 12, lineHeight: 1.4, fontFamily: "inherit",
              color: selectedQuestion === q ? "var(--color-text-info)" : "var(--color-text-secondary)",
              transition: "all 0.1s",
            }}>{q}</button>
          ))}
        </div>

        {/* Active models */}
        <div style={{
          background: "var(--color-bg-primary)", border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)", padding: "0.75rem 1rem",
        }}>
          <p style={{ fontSize: 11, fontWeight: 600, color: "var(--color-text-tertiary)", margin: "0 0 8px", textTransform: "uppercase", letterSpacing: 0.8 }}>
            Active Models
          </p>
          {selectedModels.length === 0 ? (
            <p style={{ fontSize: 11, color: "var(--color-text-danger)", margin: 0 }}>None selected — go to step 2</p>
          ) : selectedModels.map(id => {
            const m = ML_MODELS.find(x => x.id === id);
            return (
              <div key={id} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: m.color, flexShrink: 0 }} />
                <span style={{ fontSize: 10, color: "var(--color-text-secondary)", fontFamily: "'DM Mono', monospace", lineHeight: 1.4 }}>{m.model}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Chat area */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {/* Messages */}
        <div style={{
          background: "var(--color-bg-primary)", border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)", padding: "1.25rem",
          minHeight: 340, maxHeight: 500, overflowY: "auto",
          display: "flex", flexDirection: "column", gap: 18,
        }}>
          {conversation.length === 0 ? (
            <div style={{ textAlign: "center", padding: "3rem 2rem", color: "var(--color-text-tertiary)", animation: "fadeIn 0.4s ease" }}>
              <svg width="52" height="52" viewBox="0 0 52 52" fill="none" style={{ margin: "0 auto 16px", display: "block" }}>
                <rect width="52" height="52" rx="13" fill="var(--color-bg-secondary)" />
                <path d="M15 20h22M15 27h15M15 34h9" stroke="var(--color-border-secondary)" strokeWidth="1.8" strokeLinecap="round" />
                <circle cx="39" cy="34" r="7" fill="var(--color-bg-info)" stroke="var(--color-border-info)" strokeWidth="1" />
                <path d="M37 34L38.5 35.5L41 32.5" stroke="var(--color-text-info)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <p style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-secondary)", margin: "0 0 6px" }}>
                Your ML analysis engine is ready
              </p>
              <p style={{ fontSize: 12, margin: 0 }}>
                Select a quick insight from the left, or type a custom question below
              </p>
            </div>
          ) : (
            conversation.map((msg, i) => (
              <div key={i} style={{
                display: "flex", flexDirection: "column",
                alignItems: msg.role === "user" ? "flex-end" : "flex-start",
                animation: "fadeIn 0.3s ease",
              }}>
                <div style={{
                  display: "flex", alignItems: "center", gap: 6, marginBottom: 5,
                  flexDirection: msg.role === "user" ? "row-reverse" : "row",
                }}>
                  <div style={{
                    width: 24, height: 24, borderRadius: "50%",
                    background: msg.role === "user" ? "#0C1929" : "#E6F1FB",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 8, fontWeight: 700,
                    color: msg.role === "user" ? "white" : "#185FA5",
                  }}>
                    {msg.role === "user" ? "YOU" : "ML"}
                  </div>
                  <span style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>{msg.timestamp}</span>
                </div>

                <div style={{
                  maxWidth: "88%",
                  background: msg.role === "user"
                    ? "#0C1929"
                    : msg.isError ? "var(--color-bg-danger)" : "var(--color-bg-secondary)",
                  color: msg.role === "user" ? "white" : msg.isError ? "var(--color-text-danger)" : "var(--color-text-primary)",
                  padding: "0.75rem 1rem",
                  borderRadius: msg.role === "user" ? "12px 12px 3px 12px" : "12px 12px 12px 3px",
                  fontSize: 13, lineHeight: 1.65,
                }}>
                  <p style={{ margin: 0 }} dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />

                  {msg.models && msg.models.length > 0 && (
                    <div style={{ display: "flex", gap: 5, marginTop: 10, flexWrap: "wrap" }}>
                      {msg.models.map(m => (
                        <span key={m.id} style={{
                          fontSize: 9, background: m.bg, color: m.color,
                          padding: "2px 7px", borderRadius: 4,
                          fontFamily: "'DM Mono', monospace", fontWeight: 500,
                        }}>{m.model}</span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {isAnalysing && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, animation: "fadeIn 0.2s ease" }}>
              <div style={{ display: "flex", gap: 4 }}>
                {[0, 1, 2].map(i => (
                  <div key={i} style={{
                    width: 7, height: 7, borderRadius: "50%", background: "#378ADD",
                    animation: `pulse 1s ${i * 0.22}s infinite`,
                  }} />
                ))}
              </div>
              <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>
                Running ML analysis across {selectedModels.length} model{selectedModels.length !== 1 ? "s" : ""}…
              </span>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input row */}
        <div style={{
          background: "var(--color-bg-primary)", border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)", padding: "0.75rem",
        }}>
          {selectedQuestion && (
            <div style={{
              display: "flex", alignItems: "center", gap: 8, marginBottom: 8,
              padding: "6px 10px", background: "var(--color-bg-info)",
              borderRadius: "var(--border-radius-sm)", border: "0.5px solid var(--color-border-info)",
            }}>
              <span style={{ fontSize: 11, color: "var(--color-text-info)", flex: 1 }}>{selectedQuestion}</span>
              <button onClick={() => setSelectedQuestion("")} style={{
                background: "none", border: "none", cursor: "pointer",
                color: "var(--color-text-info)", fontSize: 16, padding: 0, lineHeight: 1,
              }}>×</button>
            </div>
          )}

          <div style={{ display: "flex", gap: 8 }}>
            <input
              type="text"
              placeholder="Ask anything about your SASE pipeline data… (Enter to send)"
              value={customQuestion}
              onChange={(e) => setCustomQuestion(e.target.value)}
              onKeyDown={handleKey}
              style={{
                flex: 1, padding: "0.55rem 0.9rem", fontSize: 13, fontFamily: "inherit",
                border: "0.5px solid var(--color-border-tertiary)",
                borderRadius: "var(--border-radius-md)",
                background: "var(--color-bg-secondary)",
                color: "var(--color-text-primary)", outline: "none",
              }}
            />
            <button
              onClick={runAnalysis}
              disabled={(!customQuestion.trim() && !selectedQuestion) || isAnalysing || selectedModels.length === 0}
              style={{
                padding: "0.55rem 1.4rem",
                background: isAnalysing || (!customQuestion.trim() && !selectedQuestion) ? "var(--color-bg-secondary)" : "#0C1929",
                color: isAnalysing || (!customQuestion.trim() && !selectedQuestion) ? "var(--color-text-tertiary)" : "white",
                border: "none", borderRadius: "var(--border-radius-md)", cursor: "pointer",
                fontSize: 13, fontWeight: 600, fontFamily: "inherit", transition: "all 0.15s",
                whiteSpace: "nowrap",
              }}
            >
              {isAnalysing ? "Analysing…" : "Analyse ↗"}
            </button>
          </div>

          {selectedModels.length === 0 && (
            <p style={{ fontSize: 11, color: "var(--color-text-danger)", margin: "6px 0 0" }}>
              ⚠ Select at least one ML model in step 2 before running analysis.
            </p>
          )}

          {conversation.length > 0 && (
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 6 }}>
              <button onClick={() => setConversation([])} style={{
                fontSize: 10, color: "var(--color-text-tertiary)", background: "none",
                border: "none", cursor: "pointer", fontFamily: "inherit",
              }}>Clear conversation</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedModels, setSelectedModels] = useState(["win_probability", "stage_velocity", "churn_risk"]);
  const [conversation, setConversation] = useState([]);
  const [isAnalysing, setIsAnalysing] = useState(false);

  const toggleModel = (id) => {
    setSelectedModels(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    );
  };

  const insightCount = Math.floor(conversation.length / 2);

  return (
    <div style={{ minHeight: "100vh", background: "var(--color-bg-tertiary)" }}>
      <Header vendors={VENDORS} />
      <TabBar activeTab={activeTab} setActiveTab={setActiveTab} conversationCount={insightCount} />

      <div style={{ maxWidth: 1140, margin: "0 auto", padding: "1.75rem 2rem 3rem" }}>
        <SummaryCards stats={SUMMARY_STATS} />

        {activeTab === "upload" && (
          <UploadTab
            uploadedFiles={uploadedFiles}
            setUploadedFiles={setUploadedFiles}
            setActiveTab={setActiveTab}
          />
        )}
        {activeTab === "models" && (
          <ModelsTab
            selectedModels={selectedModels}
            toggleModel={toggleModel}
            setActiveTab={setActiveTab}
          />
        )}
        {activeTab === "insights" && (
          <InsightsTab
            selectedModels={selectedModels}
            conversation={conversation}
            setConversation={setConversation}
            isAnalysing={isAnalysing}
            setIsAnalysing={setIsAnalysing}
            uploadedFiles={uploadedFiles}
          />
        )}
      </div>
    </div>
  );
}

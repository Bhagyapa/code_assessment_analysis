import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    r = json.load(f)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
base_means = [r["baseline"][m]["mean"] for m in metrics]
base_stds = [r["baseline"][m]["std"] for m in metrics]
prop_means = [r["proposed"][m]["mean"] for m in metrics]
prop_stds = [r["proposed"][m]["std"] for m in metrics]

x = np.arange(len(metrics))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, base_means, w, yerr=base_stds, capsize=4, label="Baseline (RandomForest, standard feature set)", color="#8899aa")
ax.bar(x + w/2, prop_means, w, yerr=prop_stds, capsize=4, label="Proposed (XGBoost, engineered features)", color="#a8442f")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("Score (5-fold CV mean \u00b1 std)")
ax.set_title("Baseline vs. Proposed: CV Metric Comparison")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("chart_metrics.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
data = [r["baseline"]["accuracy"]["values"], r["proposed"]["accuracy"]["values"]]
bp = ax.boxplot(data, tick_labels=["Baseline", "Proposed"], patch_artist=True, widths=0.5)
for patch, color in zip(bp["boxes"], ["#8899aa", "#a8442f"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Accuracy per fold")
ax.set_title("Per-Fold Accuracy Distribution (5-fold Stratified CV)")
fig.tight_layout()
fig.savefig("chart_boxplot.png")
plt.close(fig)

rng = np.random.default_rng(42)
diff = np.array(r["proposed"]["accuracy"]["values"]) - np.array(r["baseline"]["accuracy"]["values"])
boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
ci = r["ab_test"]["bootstrap_95ci_diff"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(boot, bins=50, color="#4472a8", alpha=0.85)
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="No difference")
ax.axvline(ci[0], color="#c0392b", linestyle=":", linewidth=1.5, label="95% CI")
ax.axvline(ci[1], color="#c0392b", linestyle=":", linewidth=1.5)
ax.set_xlabel("Proposed \u2212 Baseline mean accuracy (bootstrap resamples)")
ax.set_ylabel("Frequency")
ax.set_title("Bootstrap Distribution of Accuracy Difference")
ax.legend()
fig.tight_layout()
fig.savefig("chart_bootstrap.png")
plt.close(fig)

# Precision/recall trade-off chart -- the key operational story
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter([r["baseline"]["recall"]["mean"]], [r["baseline"]["precision"]["mean"]],
           s=180, color="#8899aa", label="Baseline", zorder=3)
ax.scatter([r["proposed"]["recall"]["mean"]], [r["proposed"]["precision"]["mean"]],
           s=180, color="#a8442f", label="Proposed", zorder=3)
ax.annotate("Baseline\n(misses >50% of\nactual cancellations)",
            (r["baseline"]["recall"]["mean"], r["baseline"]["precision"]["mean"]),
            textcoords="offset points", xytext=(15, -35), fontsize=9)
ax.annotate("Proposed\n(catches ~81% of\nactual cancellations)",
            (r["proposed"]["recall"]["mean"], r["proposed"]["precision"]["mean"]),
            textcoords="offset points", xytext=(-140, 10), fontsize=9)
ax.set_xlabel("Recall (share of true cancellations caught)")
ax.set_ylabel("Precision (share of flagged bookings that actually cancel)")
ax.set_title("Precision / Recall Trade-off")
ax.set_xlim(0.3, 1.0); ax.set_ylim(0.75, 1.0)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig("chart_pr_tradeoff.png")
plt.close(fig)


def build_html_report(results):
    metric_rows = []
    for metric_name, label in zip(metrics, labels):
        baseline_mean = results["baseline"][metric_name]["mean"]
        proposed_mean = results["proposed"][metric_name]["mean"]
        delta = proposed_mean - baseline_mean
        metric_rows.append(
            f"<tr><td>{label}</td><td>{baseline_mean:.4f}</td><td>{proposed_mean:.4f}</td><td>{delta:+.4f}</td></tr>"
        )

    html_document = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Hotel Booking Cancellation A/B Test</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f6f7fb; color: #1e2430; }}
    h1, h2 {{ margin-bottom: 0.5rem; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .card {{ background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 1.5rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; }}
    img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #dfe3eb; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border: 1px solid #dfe3eb; padding: 0.6rem; text-align: left; }}
    th {{ background: #eef2ff; }}
    .stat {{ margin: 0.5rem 0; }}
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>Hotel Booking Cancellation Prediction A/B Test</h1>

    <div class=\"card\">
      <h2>Summary</h2>
      <p class=\"stat\"><strong>Mean accuracy difference:</strong> {results['ab_test']['mean_accuracy_diff']:+.4f}</p>
      <p class=\"stat\"><strong>Paired t-statistic:</strong> {results['ab_test']['paired_t_stat']:.4f}</p>
      <p class=\"stat\"><strong>One-sided p-value:</strong> {results['ab_test']['p_value_one_sided']:.6f}</p>
      <p class=\"stat\"><strong>Bootstrap 95% CI:</strong> [{results['ab_test']['bootstrap_95ci_diff'][0]:.4f}, {results['ab_test']['bootstrap_95ci_diff'][1]:.4f}]</p>
      <p class=\"stat\"><strong>Significant at 0.05:</strong> {str(results['ab_test']['significant_at_0.05']).lower()}</p>
    </div>

    <div class=\"card\">
      <h2>CV metric comparison</h2>
      <table>
        <thead>
          <tr><th>Metric</th><th>Baseline</th><th>Proposed</th><th>Delta</th></tr>
        </thead>
        <tbody>
          {''.join(metric_rows)}
        </tbody>
      </table>
    </div>

    <div class=\"grid\">
      <div class=\"card\"><h2>Metric comparison</h2><img src=\"chart_metrics.png\" alt=\"Metric comparison chart\" /></div>
      <div class=\"card\"><h2>Per-fold accuracy</h2><img src=\"chart_boxplot.png\" alt=\"Per-fold accuracy box plot\" /></div>
      <div class=\"card\"><h2>Bootstrap difference</h2><img src=\"chart_bootstrap.png\" alt=\"Bootstrap distribution\" /></div>
      <div class=\"card\"><h2>Precision / recall trade-off</h2><img src=\"chart_pr_tradeoff.png\" alt=\"Precision recall tradeoff\" /></div>
    </div>
  </div>
</body>
</html>
"""

    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_document)


build_html_report(r)
print("charts and HTML report saved")
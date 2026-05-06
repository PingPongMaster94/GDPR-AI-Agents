import plotly.graph_objects as go
import numpy as np

def compute_score(audit_dict):
    """
    Compute average audit score from dictionary of rule matches.
    """
    if not isinstance(audit_dict, dict) or len(audit_dict) == 0:
        return 0
    return np.mean(list(audit_dict.values()))


def create_dashboard(results_df, title="Consent Form Audit Dashboard"):
    """
    Creates an interactive Plotly table showing compliance results.
    """
    # Compute numeric audit scores if not already present
    if "score" not in results_df.columns:
        results_df["score"] = results_df["audit_score"].apply(compute_score)

    # Color gradient red→green
    colors = [
        f"rgba({int(255*(1-s))}, {int(255*s)}, 100, 0.7)"
        for s in results_df["score"]
    ]

    fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=["File Name", "Audit Score", "Detected PII", "Details"],
                fill_color="#2f3e46",
                align="left",
                font=dict(color="white", size=12),
            ),
            cells=dict(
                values=[
                    results_df["file_name"],
                    [f"{round(v*100,1)} %" for v in results_df["score"]],
                    [str(x) for x in results_df.get("pii_found", [])],
                    [str(x) for x in results_df.get("audit_score", [])],
                ],
                fill_color=[colors],
                align="left",
                height=30,
            ),
        )
    ])

    fig.update_layout(
        title=title,
        height=800,
    )
    return fig

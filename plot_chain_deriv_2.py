import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from chain_deriv_2 import chain_deriv_2, sigmoid
from nested_function import chain_length_2
from derivative import deriv

def square(x: np.ndarray) -> np.ndarray:
    return x ** 2

x = np.linspace(-3, 3, 500)
chain = [square, sigmoid]

f1_of_x = square(x)
f2_of_f1 = chain_length_2(chain, x)
chain_d = chain_deriv_2(chain, x)
df1dx = deriv(square, x)
df2du = deriv(sigmoid, f1_of_x)

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=(
        "f2(f1(x)) = sigmoid(x²)",
        "Components: f1′(x) and f2′(f1(x))",
        "Chain rule derivative: f1′(x) · f2′(f1(x))",
    ),
    row_heights=[0.28, 0.32, 0.40],
)

tip = lambda name, ys: [f"{name}<br>x = {xi:.3f}<br>y = {yi:.4f}" for xi, yi in zip(x, ys)]

# --- Row 1: composed function ---
fig.add_trace(go.Scatter(
    x=x, y=f2_of_f1,
    name="sigmoid(x²)",
    line=dict(color="steelblue", width=2),
    hovertemplate="%{text}<extra></extra>",
    text=tip("sigmoid(x²)", f2_of_f1),
), row=1, col=1)

# --- Row 2: components ---
fig.add_trace(go.Scatter(
    x=x, y=df1dx,
    name="f1′(x) = 2x",
    line=dict(color="darkorange", width=2),
    hovertemplate="%{text}<extra></extra>",
    text=tip("f1′(x) = 2x", df1dx),
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=x, y=df2du,
    name="f2′(f1(x)) = sigmoid′(x²)",
    line=dict(color="mediumpurple", width=2),
    hovertemplate="%{text}<extra></extra>",
    text=tip("f2′(f1(x))", df2du),
), row=2, col=1)

# --- Row 3: chain rule derivative + shading ---
fig.add_trace(go.Scatter(
    x=x, y=np.where(chain_d >= 0, chain_d, 0),
    fill="tozeroy",
    fillcolor="rgba(0,180,0,0.12)",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip",
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=x, y=np.where(chain_d <= 0, chain_d, 0),
    fill="tozeroy",
    fillcolor="rgba(220,0,0,0.12)",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip",
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=x, y=chain_d,
    name="d/dx sigmoid(x²)",
    line=dict(color="crimson", width=2.5),
    hovertemplate="%{text}<extra></extra>",
    text=tip("d/dx sigmoid(x²)", chain_d),
), row=3, col=1)

# Annotations for peak, trough, zero
peak_i = np.argmax(chain_d)
trough_i = np.argmin(chain_d)
for i, label, ay in [(peak_i, f"max ≈ {chain_d[peak_i]:.3f}", -40),
                     (trough_i, f"min ≈ {chain_d[trough_i]:.3f}", 40)]:
    fig.add_annotation(
        x=x[i], y=chain_d[i], text=label,
        showarrow=True, arrowhead=2, ay=ay, ax=30,
        font=dict(size=10), row=3, col=1,
    )

# Zero-crossing annotation
fig.add_annotation(
    x=0, y=0, text="zero (inflection)",
    showarrow=True, arrowhead=2, ay=-40, ax=50,
    font=dict(size=10), row=3, col=1,
)

# Vertical guide lines across all panels
for xv in [x[np.argmax(chain_d)], x[np.argmin(chain_d)]]:
    for row in [1, 2, 3]:
        fig.add_vline(x=xv, line=dict(color="gray", width=1, dash="dot"), row=row, col=1)

fig.update_layout(
    title=dict(text="Chain Rule:  d/dx sigmoid(x²) = sigmoid′(x²) · 2x", font=dict(size=14)),
    hovermode="x unified",
    height=750,
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgray", borderwidth=1),
    template="plotly_white",
)

fig.update_xaxes(title_text="x", row=3, col=1)
fig.update_yaxes(title_text="sigmoid(x²)", row=1, col=1)
fig.update_yaxes(title_text="components", row=2, col=1)
fig.update_yaxes(title_text="derivative", row=3, col=1)

fig.write_html("chain_rule_plot.html")
fig.show()
print("Saved to chain_rule_plot.html")
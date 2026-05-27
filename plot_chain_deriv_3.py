import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from chain_deriv_3 import chain_deriv_3, sigmoid
from derivative import deriv

def square(x: np.ndarray) -> np.ndarray:
    return x ** 2

# Chain: square(sigmoid(square(x))) = sigmoid(x²)²
chain = [square, sigmoid, square]

x = np.linspace(-3, 3, 500)

f1_of_x   = square(x)
f2_of_f1  = sigmoid(f1_of_x)
f3_of_f2  = square(f2_of_f1)          # full composed output

df1dx  = deriv(square,   x)            # f1′(x)          ≈ 2x
df2du  = deriv(sigmoid,  f1_of_x)      # f2′(f1(x))      = sigmoid′(x²)
df3du  = deriv(square,   f2_of_f1)     # f3′(f2(f1(x)))  = 2·sigmoid(x²)

full_deriv = chain_deriv_3(chain, x)   # product of all three

peak_i   = np.argmax(full_deriv)
trough_i = np.argmin(full_deriv)

tip = lambda name, ys: [f"{name}<br>x = {xi:.3f}<br>y = {yi:.4f}" for xi, yi in zip(x, ys)]

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=(
        "f3(f2(f1(x))) = sigmoid(x²)²",
        "Components: f1′(x),  f2′(f1(x)),  f3′(f2(f1(x)))",
        "Chain rule derivative: f1′(x) · f2′(f1(x)) · f3′(f2(f1(x)))",
    ),
    row_heights=[0.28, 0.32, 0.40],
)

# --- Row 1: composed function ---
fig.add_trace(go.Scatter(
    x=x, y=f3_of_f2,
    name="sigmoid(x²)²",
    line=dict(color="steelblue", width=2),
    hovertemplate="%{text}<extra></extra>",
    text=tip("sigmoid(x²)²", f3_of_f2),
), row=1, col=1)

# --- Row 2: three components ---
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
    text=tip("f2′(f1(x)) = sigmoid′(x²)", df2du),
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=x, y=df3du,
    name="f3′(f2(f1(x))) = 2·sigmoid(x²)",
    line=dict(color="teal", width=2, dash="dash"),
    hovertemplate="%{text}<extra></extra>",
    text=tip("f3′(f2(f1(x))) = 2·sigmoid(x²)", df3du),
), row=2, col=1)

# --- Row 3: full derivative + shading ---
fig.add_trace(go.Scatter(
    x=x, y=np.where(full_deriv >= 0, full_deriv, 0),
    fill="tozeroy", fillcolor="rgba(0,180,0,0.12)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=x, y=np.where(full_deriv <= 0, full_deriv, 0),
    fill="tozeroy", fillcolor="rgba(220,0,0,0.12)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=x, y=full_deriv,
    name="d/dx sigmoid(x²)²",
    line=dict(color="crimson", width=2.5),
    hovertemplate="%{text}<extra></extra>",
    text=tip("d/dx sigmoid(x²)²", full_deriv),
), row=3, col=1)

# Annotations
fig.add_annotation(
    x=x[peak_i], y=full_deriv[peak_i],
    text=f"max ≈ {full_deriv[peak_i]:.3f}",
    showarrow=True, arrowhead=2, ay=-40, ax=30, font=dict(size=10),
    row=3, col=1,
)
fig.add_annotation(
    x=x[trough_i], y=full_deriv[trough_i],
    text=f"min ≈ {full_deriv[trough_i]:.3f}",
    showarrow=True, arrowhead=2, ay=40, ax=30, font=dict(size=10),
    row=3, col=1,
)
fig.add_annotation(
    x=0, y=0, text="zero (inflection)",
    showarrow=True, arrowhead=2, ay=-40, ax=50, font=dict(size=10),
    row=3, col=1,
)

# Vertical guide lines
for xv in [x[peak_i], x[trough_i]]:
    for row in [1, 2, 3]:
        fig.add_vline(x=xv, line=dict(color="gray", width=1, dash="dot"), row=row, col=1)

fig.update_layout(
    title=dict(
        text="Chain Rule (3 functions):  d/dx sigmoid(x²)² = 2·sigmoid(x²) · sigmoid′(x²) · 2x",
        font=dict(size=13),
    ),
    hovermode="x unified",
    height=780,
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgray", borderwidth=1),
    template="plotly_white",
)

fig.update_xaxes(title_text="x", row=3, col=1)
fig.update_yaxes(title_text="sigmoid(x²)²", row=1, col=1)
fig.update_yaxes(title_text="components", row=2, col=1)
fig.update_yaxes(title_text="derivative", row=3, col=1)

fig.write_html("chain_deriv_2_plot.html")
fig.show()
print("Saved to chain_deriv_2_plot.html")
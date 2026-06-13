import gradio as gr
import joblib
import pandas as pd
from pathlib import Path

# ── Load trained pipeline ────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "models" / "best_model.pkl"
pipeline = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
]


# ── Prediction function ──────────────────────────────────────────────────────
def predict_price(avg_income, house_age, num_rooms, num_bedrooms, population):
    """Take feature inputs and return a formatted price prediction."""
    input_df = pd.DataFrame(
        [[avg_income, house_age, num_rooms, num_bedrooms, population]],
        columns=FEATURE_COLS,
    )
    predicted = pipeline.predict(input_df)[0]
    predicted = max(predicted, 0)          # prices cannot be negative

    return (
        f"### 🏠 Estimated House Price\n"
        f"# **${predicted:,.0f}**\n\n"
        f"---\n"
        f"*Model: Lasso Regression · Test R² = 0.918*"
    )


# ── Interface ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="USA House Price Predictor",
    theme=gr.themes.Soft(),
) as interface:

    gr.Markdown(
        """
        # 🏡 USA House Price Predictor
        Estimate the market value of a house based on area-level characteristics.
        Adjust the sliders or enter values directly, then click **Predict Price**.

        > **Model:** Lasso Regression trained on 5,000 US housing records · **Test R² = 0.918**
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Input Features")

            avg_income = gr.Number(
                label="Average Area Income ($)",
                value=68_583,
                minimum=10_000,
                maximum=120_000,
                info="Average household income in the neighbourhood (USD).",
            )

            house_age = gr.Slider(
                label="Average House Age (years)",
                minimum=1.0,
                maximum=12.0,
                step=0.1,
                value=5.98,
                info="Typical age of homes in the area.",
            )

            num_rooms = gr.Slider(
                label="Average Number of Rooms",
                minimum=2.0,
                maximum=12.0,
                step=0.1,
                value=7.0,
                info="Average total number of rooms per house in the area.",
            )

            num_bedrooms = gr.Slider(
                label="Average Number of Bedrooms",
                minimum=1.0,
                maximum=8.0,
                step=0.5,
                value=4.0,
                info="Average number of bedrooms per house in the area.",
            )

            population = gr.Number(
                label="Area Population",
                value=36_164,
                minimum=100,
                maximum=80_000,
                info="Total population of the surrounding area.",
            )

            predict_btn = gr.Button("Predict Price", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Prediction")
            output = gr.Markdown(
                value="*Enter feature values and click **Predict Price** to see the estimate.*"
            )

    predict_btn.click(
        fn=predict_price,
        inputs=[avg_income, house_age, num_rooms, num_bedrooms, population],
        outputs=output,
    )

    gr.Markdown("---\n### 📋 Example Houses")
    gr.Examples(
        examples=[
            [79_545, 5.68, 7.01, 4.09, 23_087],   # example 1 – mid-range suburban
            [107_702, 9.52, 10.76, 6.5,  69_622],  # example 2 – large luxury area
            [40_000,  3.00, 4.50, 2.0,  12_000],   # example 3 – affordable area
        ],
        inputs=[avg_income, house_age, num_rooms, num_bedrooms, population],
        outputs=output,
        fn=predict_price,
        cache_examples=False,
        label="Click a row to auto-fill the inputs",
    )

    gr.Markdown(
        """
        ---
        <details>
        <summary><b>ℹ️ Feature descriptions</b></summary>

        | Feature | Description |
        |---|---|
        | **Average Area Income** | Mean annual household income in the neighbourhood (USD) |
        | **Average House Age** | Average age of houses in the area (years) |
        | **Average Number of Rooms** | Mean total room count per house |
        | **Average Number of Bedrooms** | Mean bedroom count per house |
        | **Area Population** | Total population living in the surrounding area |

        </details>
        """
    )

# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    interface.launch(share=True)

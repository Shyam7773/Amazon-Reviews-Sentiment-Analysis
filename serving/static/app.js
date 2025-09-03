document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const input = document.getElementById("input-text");
  const resultBox = document.getElementById("result");
  const predSpan = document.getElementById("prediction");
  const scoresSpan = document.getElementById("scores");

  const hBtn = document.getElementById("health-btn");
  const hDiv = document.getElementById("health");

  function pct(x) { return (x * 100).toFixed(2) + "%"; }

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = (input?.value || "").trim();
    if (!text) return;

    try {
      const res = await fetch("/predict?verbose=true", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();

      predSpan.textContent = data.prediction || "unknown";
      if (data.scores) {
        scoresSpan.textContent =
          `positive: ${pct(data.scores.positive)} | negative: ${pct(data.scores.negative)}`;
      } else {
        scoresSpan.textContent = "—";
      }
      resultBox.hidden = false;
    } catch (err) {
      console.error("Predict failed:", err);
      alert("Predict failed. Check server logs and console.");
    }
  });

  hBtn?.addEventListener("click", async () => {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      hDiv.textContent = JSON.stringify(data);
    } catch (err) {
      console.error("Health failed:", err);
      hDiv.textContent = "down";
    }
  });
});
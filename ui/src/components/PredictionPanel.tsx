import { useEffect, useState } from "react";
import { fetchApproaches, predict } from "../api";
import type { Approach, LifterDetail, PredictionOut } from "../api";
import TrajectoryChart from "./TrajectoryChart";

interface Props {
  lifter: LifterDetail;
  prediction: PredictionOut | null;
  onPredict: (p: PredictionOut) => void;
}

function kg(val: number | null): string {
  return val != null ? `${val}` : "—";
}

export default function PredictionPanel({
  lifter,
  prediction,
  onPredict,
}: Props) {
  const latestEntry = lifter.entries[lifter.entries.length - 1];
  const [targetDate, setTargetDate] = useState("");
  const [targetBW, setTargetBW] = useState(
    latestEntry?.bodyweight_kg?.toString() ?? "",
  );
  const [selectedApproach, setSelectedApproach] = useState("gradient_boosting");
  const [approaches, setApproaches] = useState<Approach[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchApproaches()
      .then((data) => {
        setApproaches(data);
        // Default to first available approach
        const firstAvailable = data.find((a) => a.available);
        if (firstAvailable) {
          setSelectedApproach(firstAvailable.name);
        }
      })
      .catch(() => {
        // Fallback if endpoint not available
        setApproaches([]);
      });
  }, []);

  async function handlePredict() {
    setLoading(true);
    setError("");
    try {
      const result = await predict(
        lifter.name,
        targetDate || undefined,
        targetBW ? parseFloat(targetBW) : undefined,
        selectedApproach,
      );
      onPredict(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  const availableApproaches = approaches.filter((a) => a.available);

  return (
    <div className="card">
      <h2>Predict Next Performance</h2>

      <div className="predict-form">
        {availableApproaches.length > 1 && (
          <label>
            Model Approach
            <select
              value={selectedApproach}
              onChange={(e) => setSelectedApproach(e.target.value)}
            >
              {availableApproaches.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.display_name}
                </option>
              ))}
            </select>
          </label>
        )}
        <label>
          Target Date
          <input
            type="date"
            value={targetDate}
            onChange={(e) => setTargetDate(e.target.value)}
            placeholder="Leave blank for next expected meet"
          />
        </label>
        <label>
          Target Bodyweight (kg)
          <input
            type="number"
            step="0.1"
            value={targetBW}
            onChange={(e) => setTargetBW(e.target.value)}
            placeholder="Leave blank for current weight"
          />
        </label>
        <button onClick={handlePredict} disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {prediction && (
        <>
          <div className="prediction-grid">
            <div className="prediction-box">
              <div className="prediction-value total-color">
                {kg(prediction.next_total_kg)}
              </div>
              <div className="prediction-label">Total (kg)</div>
            </div>
            <div className="prediction-box">
              <div className="prediction-value squat-color">
                {kg(prediction.next_squat_kg)}
              </div>
              <div className="prediction-label">Squat (kg)</div>
            </div>
            <div className="prediction-box">
              <div className="prediction-value bench-color">
                {kg(prediction.next_bench_kg)}
              </div>
              <div className="prediction-label">Bench (kg)</div>
            </div>
            <div className="prediction-box">
              <div className="prediction-value deadlift-color">
                {kg(prediction.next_deadlift_kg)}
              </div>
              <div className="prediction-label">Deadlift (kg)</div>
            </div>
          </div>

          {prediction.confidence_interval && (
            <div className="ci-text">
              95% confidence: {prediction.confidence_interval[0]} –{" "}
              {prediction.confidence_interval[1]} kg total
            </div>
          )}

          {prediction.target_date && (
            <div className="ci-text">
              Predicted for: {prediction.target_date}
              {prediction.target_bodyweight_kg
                ? ` at ${prediction.target_bodyweight_kg} kg bodyweight`
                : ""}
            </div>
          )}

          {prediction.approach && (
            <div className="ci-text">
              Model: {approaches.find((a) => a.name === prediction.approach)?.display_name ?? prediction.approach}
            </div>
          )}

          {prediction.trajectory_curve.length > 0 && (
            <div className="chart-container">
              <h3>12-Month Trajectory</h3>
              <TrajectoryChart
                curve={prediction.trajectory_curve}
                history={lifter.entries}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

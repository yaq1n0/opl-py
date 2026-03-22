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

export default function PredictionPanel({ lifter, prediction, onPredict }: Props) {
  const latestEntry = lifter.entries[lifter.entries.length - 1];
  const [targetDate, setTargetDate] = useState("");
  const [targetBW, setTargetBW] = useState(latestEntry?.bodyweight_kg?.toString() ?? "");
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
    <div className="bg-bg-card border border-border rounded-lg p-5 mb-4">
      <h2 className="text-xl font-semibold mb-4">Predict Next Performance</h2>

      <div className="flex gap-3 items-end flex-wrap mb-4">
        {availableApproaches.length > 1 && (
          <label className="flex flex-col gap-1 text-xs text-text-muted flex-1 min-w-[140px]">
            Model Approach
            <select value={selectedApproach} onChange={(e) => setSelectedApproach(e.target.value)}>
              {availableApproaches.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.display_name}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="flex flex-col gap-1 text-xs text-text-muted flex-1 min-w-[140px]">
          Target Date
          <input
            type="date"
            value={targetDate}
            onChange={(e) => setTargetDate(e.target.value)}
            placeholder="Leave blank for next expected meet"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-text-muted flex-1 min-w-[140px]">
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

      {error && (
        <div className="bg-red-400/10 border border-red-400 text-red-400 px-4 py-3 rounded-lg mb-4 text-sm">
          {error}
        </div>
      )}

      {prediction && (
        <>
          <div className="grid grid-cols-[repeat(auto-fit,minmax(120px,1fr))] gap-3">
            <div className="bg-bg-input rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-green-400">
                {kg(prediction.next_total_kg)}
              </div>
              <div className="text-xs text-text-muted uppercase tracking-wide">Total (kg)</div>
            </div>
            <div className="bg-bg-input rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-red-400">
                {kg(prediction.next_squat_kg)}
              </div>
              <div className="text-xs text-text-muted uppercase tracking-wide">Squat (kg)</div>
            </div>
            <div className="bg-bg-input rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-yellow-400">
                {kg(prediction.next_bench_kg)}
              </div>
              <div className="text-xs text-text-muted uppercase tracking-wide">Bench (kg)</div>
            </div>
            <div className="bg-bg-input rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-blue-400">
                {kg(prediction.next_deadlift_kg)}
              </div>
              <div className="text-xs text-text-muted uppercase tracking-wide">Deadlift (kg)</div>
            </div>
          </div>

          {prediction.confidence_interval && (
            <div className="text-center text-sm text-text-muted mt-3">
              95% confidence: {prediction.confidence_interval[0]} –{" "}
              {prediction.confidence_interval[1]} kg total
            </div>
          )}

          {prediction.target_date && (
            <div className="text-center text-sm text-text-muted mt-3">
              Predicted for: {prediction.target_date}
              {prediction.target_bodyweight_kg
                ? ` at ${prediction.target_bodyweight_kg} kg bodyweight`
                : ""}
            </div>
          )}

          {prediction.approach && (
            <div className="text-center text-sm text-text-muted mt-3">
              Model:{" "}
              {approaches.find((a) => a.name === prediction.approach)?.display_name ??
                prediction.approach}
            </div>
          )}

          {prediction.trajectory_curve.length > 0 && (
            <div className="mt-4">
              <h3 className="text-base font-semibold mb-2">12-Month Trajectory</h3>
              <TrajectoryChart curve={prediction.trajectory_curve} history={lifter.entries} />
            </div>
          )}
        </>
      )}
    </div>
  );
}

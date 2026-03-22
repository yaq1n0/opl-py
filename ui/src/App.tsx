import { useState } from "react";
import type { LifterDetail, PredictionOut } from "./api";
import LifterSearch from "./components/LifterSearch";
import LifterProfile from "./components/LifterProfile";
import PredictionPanel from "./components/PredictionPanel";

export default function App() {
  const [lifter, setLifter] = useState<LifterDetail | null>(null);
  const [prediction, setPrediction] = useState<PredictionOut | null>(null);

  return (
    <>
      <h1 className="text-3xl font-bold mb-1">OPL Performance Predictor</h1>
      <p className="text-text-muted text-[0.95rem] mb-6">
        Search for a powerlifter, review their competition history, and predict future performance
        using machine learning.
      </p>

      <LifterSearch
        onSelect={(l) => {
          setLifter(l);
          setPrediction(null);
        }}
      />

      {lifter && (
        <>
          <LifterProfile lifter={lifter} />
          <PredictionPanel lifter={lifter} prediction={prediction} onPredict={setPrediction} />
        </>
      )}
    </>
  );
}

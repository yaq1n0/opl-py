import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { EntryOut } from "../api";

interface Props {
  curve: [number, number][];
  history: EntryOut[];
}

export default function TrajectoryChart({ curve, history }: Props) {
  // Build chart data: historical totals + predicted trajectory
  const historicalTotals = history
    .filter((e) => e.total_kg != null)
    .map((e) => ({
      label: e.date,
      historical: e.total_kg,
      predicted: null as number | null,
    }));

  const lastHistorical =
    historicalTotals.length > 0 ? historicalTotals[historicalTotals.length - 1] : null;

  const predictedPoints = curve.map(([month, total]) => ({
    label: `+${month}mo`,
    historical: null as number | null,
    predicted: total,
  }));

  // Bridge: connect last historical to first predicted
  const bridgePoint = lastHistorical
    ? {
        label: lastHistorical.label,
        historical: lastHistorical.historical,
        predicted: lastHistorical.historical,
      }
    : null;

  const data = [
    ...historicalTotals.slice(0, -1),
    ...(bridgePoint ? [bridgePoint] : historicalTotals.slice(-1)),
    ...predictedPoints,
  ];

  // Compute Y axis domain with some padding
  const allValues = data
    .flatMap((d) => [d.historical, d.predicted])
    .filter((v): v is number => v != null);
  const minY = Math.floor(Math.min(...allValues) * 0.95);
  const maxY = Math.ceil(Math.max(...allValues) * 1.05);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
        <CartesianGrid stroke="#2e3347" strokeDasharray="3 3" />
        <XAxis dataKey="label" tick={{ fill: "#8b8fa3", fontSize: 11 }} tickLine={false} />
        <YAxis
          domain={[minY, maxY]}
          tick={{ fill: "#8b8fa3", fontSize: 11 }}
          tickLine={false}
          unit=" kg"
        />
        <Tooltip
          contentStyle={{
            background: "#1a1d27",
            border: "1px solid #2e3347",
            borderRadius: 8,
            fontSize: 13,
          }}
          formatter={(value: number) => [`${value} kg`, ""]}
        />
        <Line
          type="monotone"
          dataKey="historical"
          stroke="#4ade80"
          strokeWidth={2}
          dot={{ r: 3, fill: "#4ade80" }}
          connectNulls={false}
          name="Historical"
        />
        <Line
          type="monotone"
          dataKey="predicted"
          stroke="#6c63ff"
          strokeWidth={2}
          strokeDasharray="6 3"
          dot={{ r: 3, fill: "#6c63ff" }}
          connectNulls={false}
          name="Predicted"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

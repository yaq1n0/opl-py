import type { LifterDetail } from "../api";

interface Props {
  lifter: LifterDetail;
}

function kg(val: number | null): string {
  return val != null ? `${val}` : "—";
}

export default function LifterProfile({ lifter }: Props) {
  return (
    <div className="bg-bg-card border border-border rounded-lg p-5 mb-4">
      <h2 className="text-xl font-semibold mb-4">{lifter.name}</h2>

      <div className="grid grid-cols-[repeat(auto-fit,minmax(140px,1fr))] gap-3 mb-4">
        <div className="bg-bg-input rounded-lg p-3 text-center">
          <div className="text-xl font-bold text-accent">{lifter.competition_count}</div>
          <div className="text-xs text-text-muted uppercase tracking-wide">Competitions</div>
        </div>
        <div className="bg-bg-input rounded-lg p-3 text-center">
          <div className="text-xl font-bold text-green-400">{kg(lifter.best_total_kg)}</div>
          <div className="text-xs text-text-muted uppercase tracking-wide">Best Total</div>
        </div>
        <div className="bg-bg-input rounded-lg p-3 text-center">
          <div className="text-xl font-bold text-red-400">{kg(lifter.best_squat_kg)}</div>
          <div className="text-xs text-text-muted uppercase tracking-wide">Best Squat</div>
        </div>
        <div className="bg-bg-input rounded-lg p-3 text-center">
          <div className="text-xl font-bold text-yellow-400">{kg(lifter.best_bench_kg)}</div>
          <div className="text-xs text-text-muted uppercase tracking-wide">Best Bench</div>
        </div>
        <div className="bg-bg-input rounded-lg p-3 text-center">
          <div className="text-xl font-bold text-blue-400">{kg(lifter.best_deadlift_kg)}</div>
          <div className="text-xs text-text-muted uppercase tracking-wide">Best Deadlift</div>
        </div>
      </div>

      <h3 className="text-base font-semibold mb-2">Competition History</h3>
      <div className="overflow-x-auto">
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Meet</th>
              <th>Equip</th>
              <th>BW</th>
              <th>Squat</th>
              <th>Bench</th>
              <th>Deadlift</th>
              <th>Total</th>
              <th>Dots</th>
              <th>Place</th>
            </tr>
          </thead>
          <tbody>
            {lifter.entries.map((e, i) => (
              <tr key={i}>
                <td>{e.date}</td>
                <td>{e.meet_name}</td>
                <td>{e.equipment}</td>
                <td>{kg(e.bodyweight_kg)}</td>
                <td className="text-red-400">{kg(e.best3_squat_kg)}</td>
                <td className="text-yellow-400">{kg(e.best3_bench_kg)}</td>
                <td className="text-blue-400">{kg(e.best3_deadlift_kg)}</td>
                <td className="text-green-400">{kg(e.total_kg)}</td>
                <td>{kg(e.dots)}</td>
                <td>{e.place}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

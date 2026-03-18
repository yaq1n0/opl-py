import type { LifterDetail } from "../api";

interface Props {
  lifter: LifterDetail;
}

function kg(val: number | null): string {
  return val != null ? `${val}` : "—";
}

export default function LifterProfile({ lifter }: Props) {
  return (
    <div className="card">
      <h2>{lifter.name}</h2>

      <div className="stats-grid">
        <div className="stat-box">
          <div className="stat-value">{lifter.competition_count}</div>
          <div className="stat-label">Competitions</div>
        </div>
        <div className="stat-box">
          <div className="stat-value total-color">{kg(lifter.best_total_kg)}</div>
          <div className="stat-label">Best Total</div>
        </div>
        <div className="stat-box">
          <div className="stat-value squat-color">{kg(lifter.best_squat_kg)}</div>
          <div className="stat-label">Best Squat</div>
        </div>
        <div className="stat-box">
          <div className="stat-value bench-color">{kg(lifter.best_bench_kg)}</div>
          <div className="stat-label">Best Bench</div>
        </div>
        <div className="stat-box">
          <div className="stat-value deadlift-color">{kg(lifter.best_deadlift_kg)}</div>
          <div className="stat-label">Best Deadlift</div>
        </div>
      </div>

      <h3>Competition History</h3>
      <div className="table-scroll">
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
                <td className="squat-color">{kg(e.best3_squat_kg)}</td>
                <td className="bench-color">{kg(e.best3_bench_kg)}</td>
                <td className="deadlift-color">{kg(e.best3_deadlift_kg)}</td>
                <td className="total-color">{kg(e.total_kg)}</td>
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

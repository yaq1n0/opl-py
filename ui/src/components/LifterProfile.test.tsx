import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import LifterProfile from "./LifterProfile";
import type { LifterDetail } from "../api";

const mockLifter: LifterDetail = {
  name: "Ray Williams",
  competition_count: 10,
  best_total_kg: 1000,
  best_squat_kg: 490,
  best_bench_kg: 260,
  best_deadlift_kg: 370,
  entries: [
    {
      date: "2023-10-01",
      federation: "USAPL",
      meet_name: "USAPL Raw Nationals",
      equipment: "Raw",
      event: "SBD",
      bodyweight_kg: 145,
      weight_class_kg: "140+",
      best3_squat_kg: 490,
      best3_bench_kg: 260,
      best3_deadlift_kg: 370,
      total_kg: 1000,
      dots: 550.5,
      wilks: 480.2,
      place: "1",
      age: 32,
      tested: true,
    },
  ],
};

describe("LifterProfile", () => {
  it("renders the lifter name", () => {
    render(<LifterProfile lifter={mockLifter} />);
    expect(screen.getByText("Ray Williams")).toBeInTheDocument();
  });

  it("renders competition count", () => {
    render(<LifterProfile lifter={mockLifter} />);
    expect(screen.getByText("10")).toBeInTheDocument();
  });

  it("renders best lift values", () => {
    render(<LifterProfile lifter={mockLifter} />);
    // Values appear in both the stat card and the table row, so use getAllByText
    expect(screen.getAllByText("1000").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("490").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("260").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("370").length).toBeGreaterThanOrEqual(1);
  });

  it("renders em dash for null lift values", () => {
    const lifterWithNulls: LifterDetail = {
      ...mockLifter,
      best_squat_kg: null,
      best_bench_kg: null,
      best_deadlift_kg: null,
      entries: [],
    };
    render(<LifterProfile lifter={lifterWithNulls} />);
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(3);
  });

  it("renders competition history rows", () => {
    render(<LifterProfile lifter={mockLifter} />);
    expect(screen.getByText("USAPL Raw Nationals")).toBeInTheDocument();
    expect(screen.getByText("2023-10-01")).toBeInTheDocument();
    expect(screen.getByText("Raw")).toBeInTheDocument();
  });

  it("renders empty table when entries is empty", () => {
    const lifterNoEntries: LifterDetail = { ...mockLifter, entries: [] };
    render(<LifterProfile lifter={lifterNoEntries} />);
    expect(screen.getByText("Competition History")).toBeInTheDocument();
    expect(screen.queryByText("USAPL Raw Nationals")).not.toBeInTheDocument();
  });
});

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import LifterSearch from "./LifterSearch";

vi.mock("../api", () => ({
  searchLifters: vi.fn(),
  getLifter: vi.fn(),
}));

import { searchLifters, getLifter } from "../api";

const mockSearchLifters = vi.mocked(searchLifters);
const mockGetLifter = vi.mocked(getLifter);

const mockResults = [
  { name: "Ray Williams", competition_count: 10, best_total_kg: 1000 },
  { name: "John Haack", competition_count: 8, best_total_kg: 855 },
];

const mockDetail = {
  name: "Ray Williams",
  competition_count: 10,
  best_total_kg: 1000,
  best_squat_kg: 490,
  best_bench_kg: 260,
  best_deadlift_kg: 370,
  entries: [],
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("LifterSearch", () => {
  it("renders the search input", () => {
    render(<LifterSearch onSelect={vi.fn()} />);
    expect(screen.getByPlaceholderText(/search lifters/i)).toBeInTheDocument();
  });

  it("does not search when query is less than 2 characters", async () => {
    const user = userEvent.setup();
    render(<LifterSearch onSelect={vi.fn()} />);
    await user.type(screen.getByPlaceholderText(/search lifters/i), "R");
    // Wait past the debounce window and confirm no API call was made
    await new Promise((r) => setTimeout(r, 350));
    expect(mockSearchLifters).not.toHaveBeenCalled();
  });

  it("shows search results after debounce", async () => {
    mockSearchLifters.mockResolvedValue(mockResults);
    const user = userEvent.setup();

    render(<LifterSearch onSelect={vi.fn()} />);
    await user.type(screen.getByPlaceholderText(/search lifters/i), "Ray");

    await waitFor(
      () => {
        expect(screen.getByText("Ray Williams")).toBeInTheDocument();
        expect(screen.getByText("John Haack")).toBeInTheDocument();
      },
      { timeout: 2000 },
    );
  });

  it("calls onSelect with lifter detail when a result is clicked", async () => {
    mockSearchLifters.mockResolvedValue(mockResults);
    mockGetLifter.mockResolvedValue(mockDetail);
    const onSelect = vi.fn();
    const user = userEvent.setup();

    render(<LifterSearch onSelect={onSelect} />);
    await user.type(screen.getByPlaceholderText(/search lifters/i), "Ray");

    await waitFor(() => expect(screen.getByText("Ray Williams")).toBeInTheDocument(), {
      timeout: 2000,
    });
    await user.click(screen.getByText("Ray Williams"));

    await waitFor(() => expect(onSelect).toHaveBeenCalledWith(mockDetail));
  });

  it("shows error message when search fails", async () => {
    mockSearchLifters.mockRejectedValue(new Error("Network error"));
    const user = userEvent.setup();

    render(<LifterSearch onSelect={vi.fn()} />);
    await user.type(screen.getByPlaceholderText(/search lifters/i), "Ray");

    await waitFor(() => expect(screen.getByText("Network error")).toBeInTheDocument(), {
      timeout: 2000,
    });
  });

  it("displays competition count and total in results", async () => {
    mockSearchLifters.mockResolvedValue(mockResults);
    const user = userEvent.setup();

    render(<LifterSearch onSelect={vi.fn()} />);
    await user.type(screen.getByPlaceholderText(/search lifters/i), "Ray");

    await waitFor(
      () => {
        expect(screen.getByText(/10 meets/)).toBeInTheDocument();
        expect(screen.getByText(/1000 kg total/)).toBeInTheDocument();
      },
      { timeout: 2000 },
    );
  });
});

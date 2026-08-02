import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

const epoch = {
  event_type: "epoch" as const,
  epoch: 2,
  timestamp: "2026-08-02T09:00:00Z",
  train_loss: 0.4,
  val_loss: 0.5,
  pixel_accuracy: 0.8,
  mean_iou: 0.6,
  per_class: {
    big_rock: { support: 4, predicted: 4, true_positive: 3, false_positive: 1, false_negative: 1, iou: 0.6, dice_f1: 0.75, precision: 0.75, recall: 0.75 }
  }
};

const run = {
  run_id: "review-run",
  experiment_name: "Weighted U-Net",
  hypothesis: "Class weighting improves big-rock recall.",
  researcher_notes: "Inspect false positives before promotion.",
  tags: ["weighted", "ablation"],
  status: "completed" as const,
  started_at: "2026-08-02T09:00:00Z",
  protocol_valid: true,
  failed_gates: [],
  split_role: "crowdsourced_validation",
  dataset_manifest_sha256: "a".repeat(64),
  random_seeds: { torch: 17 },
  model: "Unet",
  encoder: "resnet34",
  latest_epoch: epoch,
  summary: { status: "completed" as const, best_epoch: 2, best_validation_mean_iou: 0.6 }
};

const detail = {
  metadata: {
    run_id: "review-run",
    experiment_name: "Weighted U-Net",
    status: "completed" as const,
    tags: ["weighted"],
    provenance: { dataset_name: "AI4Mars", dataset_version: "0.6", dataset_manifest_sha256: "a".repeat(64), split_manifest_hashes: {}, split_role: "crowdsourced_validation", protocol: { valid: true, failed_gates: [] } },
    model: { name: "Unet", encoder: "resnet34" },
    training: { optimizer: "AdamW", loss: "CrossEntropyLoss" },
    artifact_refs: []
  },
  summary: run.summary,
  metrics: [epoch],
  system_metrics: []
};

const overview = {
  active_run: null,
  best_protocol_valid_validation_run: null,
  recent_runs: [],
  failed_runs: [],
  system_health: null,
  expert_test_locked: true
};

const provenance = {
  available: true,
  dataset_name: "AI4Mars",
  dataset_version: "0.6",
  gates: [{ name: "grouped_split_isolation", passed: true, detail: "No overlap." }],
  issues: []
};

describe("App", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn((input: string) => {
      const payload = input.includes("/overview") ? overview
        : input.includes("/provenance") ? provenance
          : input.includes("/runs/review-run/samples") ? { available: true, total: 1, offset: 0, limit: 4, available_splits: ["validation"], samples: [{ sample_id: "sample-1", split: "validation", image_iou: 0.2, assets: {} }] }
            : input.includes("/runs/review-run") ? detail
              : { runs: [run], warnings: [] };
      return Promise.resolve({ ok: true, json: () => Promise.resolve(payload) });
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("shows the sealed-test indicator and honest empty state", async () => {
    render(<App />);

    expect(await screen.findByText("EXPERT TEST SET LOCKED")).toBeInTheDocument();
    expect(await screen.findByText("No eligible benchmark")).toBeInTheDocument();
  });

  it("shows registry metadata needed for experiment review", async () => {
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Experiments" }));

    expect(await screen.findByRole("columnheader", { name: "Hypothesis / notes" })).toBeInTheDocument();
    expect(screen.getByText("Class weighting improves big-rock recall.")).toBeInTheDocument();
    expect(screen.getByText("Inspect false positives before promotion.")).toBeInTheDocument();
    expect(screen.getByText("torch=17")).toBeInTheDocument();
  });

  it("loads available workbench splits and sends the selected split filter", async () => {
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Workbench" }));
    const splitSelect = await screen.findByRole("combobox", { name: "Filter samples by split" });
    expect(screen.getByRole("option", { name: "validation" })).toBeInTheDocument();
    fireEvent.change(splitSelect, { target: { value: "validation" } });

    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input).includes("split=validation"))).toBe(true));
  });
});
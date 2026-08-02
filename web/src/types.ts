export type RunStatus = "queued" | "running" | "completed" | "failed" | "interrupted" | "invalid";

export interface ProtocolRecord {
  valid: boolean;
  failed_gates: string[];
  notes?: string[];
}

export interface ClassMetrics {
  support: number;
  predicted: number;
  true_positive: number;
  false_positive: number;
  false_negative: number;
  iou: number | null;
  dice_f1: number | null;
  precision: number | null;
  recall: number | null;
}

export interface EpochEvent {
  event_type: "epoch";
  epoch: number;
  timestamp: string;
  train_loss?: number;
  val_loss?: number;
  pixel_accuracy?: number;
  mean_iou?: number;
  per_class?: Record<string, ClassMetrics>;
  learning_rate?: number;
  epoch_duration_seconds?: number;
  confusion_matrix?: number[][];
  [key: string]: unknown;
}

export interface BatchEvent {
  event_type: "batch";
  epoch: number;
  batch: number;
  total_batches: number;
  timestamp: string;
  loss: number;
  smoothed_loss?: number;
  throughput_samples_per_second?: number;
  eta_seconds?: number;
  [key: string]: unknown;
}

export interface SystemEvent {
  event_type: "system";
  timestamp: string;
  cpu_percent?: number;
  ram_percent?: number;
  ram_used_bytes?: number;
  disk_read_bytes?: number;
  disk_write_bytes?: number;
  gpu_utilization_percent?: number;
  gpu_temperature_celsius?: number;
  gpu_memory_used_bytes?: number;
  gpu_memory_total_bytes?: number;
  gpu_memory_allocated_bytes?: number;
  gpu_memory_reserved_bytes?: number;
  gpu_available: boolean;
  [key: string]: unknown;
}

export interface RunSummary {
  status: RunStatus;
  best_epoch?: number;
  best_validation_mean_iou?: number;
  failure_reason?: string;
  [key: string]: unknown;
}

export interface RunCard {
  run_id: string;
  experiment_name: string;
  hypothesis?: string | null;
  researcher_notes?: string | null;
  tags: string[];
  status: RunStatus;
  started_at?: string | null;
  ended_at?: string | null;
  protocol_valid: boolean;
  failed_gates: string[];
  split_role: string;
  dataset_version?: string;
  dataset_manifest_sha256?: string;
  git_commit?: string | null;
  git_branch?: string | null;
  random_seeds?: Record<string, number>;
  model?: string;
  encoder?: string | null;
  latest_epoch?: EpochEvent | null;
  summary?: RunSummary | null;
  legacy?: boolean;
}

export interface RunDetail {
  metadata: {
    run_id: string;
    experiment_name: string;
    hypothesis?: string;
    tags?: string[];
    status: RunStatus;
    researcher_notes?: string;
    provenance: {
      dataset_name: string;
      dataset_version: string;
      source_record?: string;
      dataset_manifest_sha256: string;
      split_manifest_hashes: Record<string, string>;
      split_role: string;
      protocol: ProtocolRecord;
      git_commit?: string;
      git_branch?: string;
      git_dirty?: boolean;
    };
    model: { name: string; encoder?: string; input_resolution?: [number, number] };
    training: { optimizer: string; loss: string; learning_rate?: number; batch_size?: number; epochs?: number };
    environment?: { python?: string; pytorch?: string; cuda?: string; cudnn?: string; gpu?: string; cpu?: string; memory_total_bytes?: number };
    artifact_refs?: Array<{ path: string; kind: string; description?: string }>;
    legacy?: boolean;
  };
  summary: RunSummary | null;
  metrics: Array<EpochEvent | BatchEvent>;
  system_metrics: SystemEvent[];
}

export interface Overview {
  active_run: RunCard | null;
  best_protocol_valid_validation_run: RunCard | null;
  recent_runs: RunCard[];
  failed_runs: RunCard[];
  system_health: SystemEvent | null;
  expert_test_locked: boolean;
  ranking_warning?: string | null;
}

export interface Provenance {
  available: boolean;
  dataset_name?: string;
  dataset_version?: string;
  source_record?: string;
  manifest_path?: string;
  manifest_sha256?: string;
  pair_counts?: Record<string, number>;
  label_roles?: Record<string, number>;
  label_schemes?: Record<string, number>;
  class_pixel_counts?: Record<string, number>;
  unmatched_or_excluded?: Record<string, number>;
  grouping_keys?: string[];
  splits?: Record<string, { rows: number; source_groups: number; sequence_groups: number; sha256: string }>;
  gates: Array<{ name: string; passed: boolean; detail: string }>;
  issues: string[];
}

export interface SampleRecord {
  sample_id: string;
  split: string;
  image_iou?: number;
  loss?: number;
  uncertainty?: number;
  big_rock_false_negative?: boolean;
  big_rock_to_soil?: boolean;
  assets: Record<string, string>;
  synthetic_demo?: boolean;
}
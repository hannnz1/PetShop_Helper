-- Apply once after ch09-observability.sql; additive Task 5 migration.
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS eval_runs (
  run_id VARCHAR(64) NOT NULL PRIMARY KEY,
  dataset_hash VARCHAR(64) NOT NULL,
  strategy VARCHAR(32) NOT NULL,
  git_sha VARCHAR(40) NOT NULL,
  model_name VARCHAR(128) NOT NULL,
  started_at DATETIME NOT NULL,
  ended_at DATETIME NOT NULL,
  sample_count INT UNSIGNED NOT NULL,
  status VARCHAR(32) NOT NULL,
  metrics JSON NOT NULL,
  denominators JSON NOT NULL,
  report_path VARCHAR(512) NOT NULL,
  reason VARCHAR(256) NULL,
  KEY idx_eval_comparable (dataset_hash, strategy, started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Run once on an existing Chapter 8 database before enabling Chapter 9 usage.
-- No prompt, reply, tool arguments or secret is added to the usage table.
SET NAMES utf8mb4;

ALTER TABLE tool_audit_logs
  ADD COLUMN turn_id VARCHAR(36) NULL AFTER conversation_id,
  ADD KEY idx_tool_audit_turn (conversation_id, turn_id);

CREATE TABLE model_usage_events (
  run_id VARCHAR(36) NOT NULL,
  conversation_id BIGINT UNSIGNED NOT NULL,
  turn_id VARCHAR(36) NOT NULL,
  intent VARCHAR(64) NOT NULL,
  model_name VARCHAR(128) NOT NULL,
  input_tokens INT UNSIGNED NULL,
  output_tokens INT UNSIGNED NULL,
  usage_status ENUM('available','unavailable') NOT NULL,
  turn_status ENUM('completed','failed','interrupted') NOT NULL,
  is_estimated TINYINT NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id),
  KEY idx_usage_turn (conversation_id, turn_id),
  KEY idx_usage_daily (created_at, intent, model_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Chapter 4 low-confidence question pool. Apply only after backing up the app database.
SET NAMES utf8mb4;

CREATE TABLE low_confidence_questions (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NULL,
  raw_question TEXT NOT NULL,
  source ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL,
  reason TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_low_confidence_conversation (conversation_id),
  KEY idx_low_confidence_source (source),
  CONSTRAINT fk_low_confidence_conversation FOREIGN KEY (conversation_id)
    REFERENCES conversations (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE faith_cases (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  eval_id VARCHAR(16) NOT NULL,
  bucket VARCHAR(24) NOT NULL,
  query VARCHAR(512) NOT NULL,
  strategy VARCHAR(24) NOT NULL DEFAULT 'hybrid_rerank',
  answer TEXT NOT NULL,
  reason TEXT NOT NULL,
  citations JSON NULL,
  judge_model VARCHAR(64) NULL,
  status ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决',
  seen_count INT UNSIGNED NOT NULL DEFAULT 1,
  first_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  resolution VARCHAR(300) NULL,
  resolved_at DATETIME NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_faith_eval_id (eval_id),
  KEY idx_faith_status (status),
  KEY idx_faith_last_seen_at (last_seen_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

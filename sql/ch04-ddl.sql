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

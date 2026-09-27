-- Additive Chapter 9 data-flywheel schema. Apply once after ch04-ddl.sql.
ALTER TABLE low_confidence_questions
  ADD COLUMN source_ref VARCHAR(128) NULL,
  ADD UNIQUE KEY uq_low_confidence_source_ref (source_ref);

CREATE TABLE canonical_questions (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  canonical_question VARCHAR(512) NOT NULL,
  canonical_key CHAR(64) NULL,
  draft_answer TEXT NULL,
  status ENUM('pending_review','deferred','rejected','approved','approved_pending_vector') NOT NULL DEFAULT 'pending_review',
  approved_answer TEXT NULL,
  category VARCHAR(64) NULL,
  merged_into_id BIGINT UNSIGNED NULL,
  knowledge_chunk_id BIGINT UNSIGNED NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_canonical_key (canonical_key),
  KEY idx_canonical_status (status),
  KEY idx_canonical_knowledge_chunk (knowledge_chunk_id),
  KEY idx_canonical_merged_into (merged_into_id),
  CONSTRAINT fk_canonical_knowledge_chunk FOREIGN KEY (knowledge_chunk_id)
    REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_canonical_merged_into FOREIGN KEY (merged_into_id)
    REFERENCES canonical_questions (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE canonical_occurrences (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  canonical_id BIGINT UNSIGNED NOT NULL,
  raw_question_id BIGINT UNSIGNED NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_canonical_occurrence_raw (raw_question_id),
  KEY idx_canonical_occurrences_canonical (canonical_id),
  CONSTRAINT fk_canonical_occurrence_canonical FOREIGN KEY (canonical_id)
    REFERENCES canonical_questions (id),
  CONSTRAINT fk_canonical_occurrence_raw FOREIGN KEY (raw_question_id)
    REFERENCES low_confidence_questions (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE flywheel_review_actions (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  canonical_id BIGINT UNSIGNED NOT NULL,
  request_id VARCHAR(64) NOT NULL,
  action ENUM('reject','defer','merge','approve','publish') NOT NULL,
  reason TEXT NULL,
  approved_answer TEXT NULL,
  category VARCHAR(64) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_flywheel_review_request (request_id),
  KEY idx_flywheel_review_canonical (canonical_id),
  CONSTRAINT fk_flywheel_review_canonical FOREIGN KEY (canonical_id)
    REFERENCES canonical_questions (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

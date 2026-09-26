-- Chapter 7: additive, non-destructive context anchors and append-only summary segments.
ALTER TABLE conversations ADD COLUMN summary TEXT NULL;
ALTER TABLE conversations ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL;
ALTER TABLE conversations ADD COLUMN layer1_from_msg_id BIGINT UNSIGNED NULL;

CREATE TABLE conversation_summaries (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  seq INT UNSIGNED NOT NULL,
  from_msg_id BIGINT UNSIGNED NOT NULL,
  upto_msg_id BIGINT UNSIGNED NOT NULL,
  content TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_conversation_summaries_seq (conversation_id, seq),
  CONSTRAINT fk_conversation_summaries_conversation
    FOREIGN KEY (conversation_id) REFERENCES conversations (id),
  CONSTRAINT ck_conversation_summaries_range CHECK (from_msg_id <= upto_msg_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Append-only early context segments';

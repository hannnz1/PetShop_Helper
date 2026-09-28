CREATE TABLE IF NOT EXISTS topic_classifications (
 id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
 question_id BIGINT UNSIGNED NOT NULL,
 labels JSON NOT NULL,
 model_version VARCHAR(64) NOT NULL,
 taxonomy_hash VARCHAR(64) NOT NULL,
 threshold DOUBLE NOT NULL,
 input_hash VARCHAR(64) NOT NULL,
 run_id VARCHAR(64) NOT NULL,
 classified_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE KEY uq_topic_question (question_id),
 KEY idx_topic_classified (classified_at, id),
 CONSTRAINT fk_topic_question FOREIGN KEY (question_id) REFERENCES low_confidence_questions(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS topic_classification_runs (
 run_id VARCHAR(64) NOT NULL PRIMARY KEY,
 model_version VARCHAR(64) NULL,
 started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
 finished_at DATETIME NULL,
 pending_count INT UNSIGNED NOT NULL,
 success_count INT UNSIGNED NOT NULL,
 failed_count INT UNSIGNED NOT NULL,
 status VARCHAR(24) NOT NULL,
 report_path VARCHAR(255) NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

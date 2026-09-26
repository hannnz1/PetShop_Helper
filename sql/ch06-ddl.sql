-- Chapter 6 demo order source and confirmed refund applications.
SET NAMES utf8mb4;

CREATE TABLE sample_orders (
  order_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  status VARCHAR(64) NOT NULL,
  product VARCHAR(255) NOT NULL,
  amount DECIMAL(10,2) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (order_id),
  KEY idx_sample_orders_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Chapter 6 demonstration orders only';

CREATE TABLE refund_requests (
  refund_no VARCHAR(32) NOT NULL,
  request_id VARCHAR(64) NOT NULL,
  conversation_id BIGINT UNSIGNED NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  order_id VARCHAR(64) NOT NULL,
  reason ENUM('七天无理由','质量问题','发错货','不想要了','其他') NOT NULL,
  status ENUM('待人工审核') NOT NULL DEFAULT '待人工审核',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (refund_no),
  UNIQUE KEY uq_refund_requests_request_id (request_id),
  KEY idx_refund_requests_conversation (conversation_id),
  KEY idx_refund_requests_order (order_id),
  CONSTRAINT fk_refund_requests_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id),
  CONSTRAINT fk_refund_requests_order FOREIGN KEY (order_id) REFERENCES sample_orders (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Confirmed demo refund applications, no payment action';

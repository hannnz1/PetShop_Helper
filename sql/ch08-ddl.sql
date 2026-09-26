-- Chapter 8 tool-call audit. Deliberately no foreign key to conversation.
SET NAMES utf8mb4;

CREATE TABLE tool_audit_logs (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NULL,
  tool_call_id VARCHAR(64) NULL,
  tool_name VARCHAR(128) NOT NULL,
  tool_source ENUM('builtin','mcp') NOT NULL,
  mcp_server VARCHAR(64) NULL,
  arguments JSON NULL,
  result_summary TEXT NULL,
  status ENUM('成功','失败','超时','校验拦下','权限拒绝') NOT NULL,
  error_message VARCHAR(512) NULL,
  retry_count TINYINT UNSIGNED NOT NULL DEFAULT 0,
  duration_ms INT UNSIGNED NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  KEY idx_tool_name (tool_name),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

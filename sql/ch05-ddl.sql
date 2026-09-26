-- Existing tickets have NULL request_id; MySQL permits multiple NULLs in a unique key.
ALTER TABLE tickets
  ADD COLUMN request_id VARCHAR(64) NULL COMMENT 'Client-generated idempotency key',
  ADD UNIQUE KEY uq_tickets_request_id (request_id);

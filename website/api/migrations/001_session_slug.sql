-- Add a human-readable slug to a session, so a workshop can be joined from a
-- plain link (…/workshop/?w=namur-2026) instead of a typed four-character code.
--
-- Apply with:  mysql --defaults-file=~/.negawatt_ws.cnf < 001_session_slug.sql
-- MySQL 8.0 has no ADD COLUMN IF NOT EXISTS, so this is guarded by hand and is
-- safe to re-run.

SET @has_col := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ws_sessions' AND COLUMN_NAME = 'slug'
);
SET @sql := IF(@has_col = 0,
  'ALTER TABLE ws_sessions ADD COLUMN slug VARCHAR(64) NULL AFTER code',
  'SELECT "column slug already present" AS note');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @has_idx := (
  SELECT COUNT(*) FROM information_schema.STATISTICS
   WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ws_sessions' AND INDEX_NAME = 'uq_slug'
);
SET @sql := IF(@has_idx = 0,
  'ALTER TABLE ws_sessions ADD UNIQUE KEY uq_slug (slug)',
  'SELECT "index uq_slug already present" AS note');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

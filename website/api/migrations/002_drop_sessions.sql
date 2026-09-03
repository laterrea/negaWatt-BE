-- Remove the session layer: a group now belongs to a topic directly, and the
-- reveal screen selects groups by date instead of by session code.
--
-- Apply with:  mysql --defaults-file=~/.negawatt_ws.cnf < 002_drop_sessions.sql
--
-- MySQL 8.0 has no ADD/DROP ... IF EXISTS for columns and constraints, so every
-- step is guarded by hand and the whole file is safe to re-run. Answers are kept:
-- each group inherits the topic of the session it belonged to.

-- 1. ws_groups.topic ----------------------------------------------------------
SET @has_col := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ws_groups' AND COLUMN_NAME = 'topic'
);
SET @sql := IF(@has_col = 0,
  'ALTER TABLE ws_groups ADD COLUMN topic VARCHAR(64) NOT NULL DEFAULT "" AFTER id',
  'SELECT "column topic already present" AS note');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- 2. carry the topic over from the sessions, while they are still there --------
SET @has_sessions := (
  SELECT COUNT(*) FROM information_schema.TABLES
   WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ws_sessions'
);
SET @sql := IF(@has_sessions = 1,
  'UPDATE ws_groups g JOIN ws_sessions s ON s.code = g.session_code
      SET g.topic = s.topic WHERE g.topic = ""',
  'SELECT "no ws_sessions to migrate from" AS note');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- 3. the foreign key and the unique name index both mention session_code -------
SET @fk := (
  SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS
   WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ws_groups'
     AND CONSTRAINT_NAME = 'fk_group_session'
);
SET @sql := IF(@fk = 1,
  'ALTER TABLE ws_groups DROP FOREIGN KEY fk_group_session',
  'SELECT "no fk_group_session" AS note');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @idx := (
  SELECT COUNT(DISTINCT INDEX_NAME) FROM information_schema.STATISTICS
   WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ws_groups'
     AND INDEX_NAME = 'uq_session_name'
);
SET @sql := IF(@idx = 1,
  'ALTER TABLE ws_groups DROP INDEX uq_session_name',
  'SELECT "no uq_session_name" AS note');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- 4. drop session_code, add the index the new queries use ----------------------
SET @has_col := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ws_groups'
     AND COLUMN_NAME = 'session_code'
);
SET @sql := IF(@has_col = 1,
  'ALTER TABLE ws_groups DROP COLUMN session_code',
  'SELECT "session_code already gone" AS note');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @idx := (
  SELECT COUNT(DISTINCT INDEX_NAME) FROM information_schema.STATISTICS
   WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ws_groups'
     AND INDEX_NAME = 'idx_topic_created'
);
SET @sql := IF(@idx = 0,
  'ALTER TABLE ws_groups ADD KEY idx_topic_created (topic, created_at)',
  'SELECT "idx_topic_created already present" AS note');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- 5. and the sessions themselves ----------------------------------------------
DROP TABLE IF EXISTS ws_sessions;

-- 6. match schema.sql exactly: the empty default was only there to let step 1
--    add the column to populated rows. Unconditional, and idempotent.
ALTER TABLE ws_groups MODIFY COLUMN topic VARCHAR(64) NOT NULL;

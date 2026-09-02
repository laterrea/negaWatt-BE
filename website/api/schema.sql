-- =============================================================================
-- negaWatt-BE workshop — MySQL schema
--
-- Apply with (no root needed once setup-negawatt-workshop-db.sh has run):
--     mysql --defaults-file=~/.negawatt_ws.cnf < schema.sql
--
-- Every statement is IF NOT EXISTS, so re-running is safe.
--
-- Table names carry a ws_ prefix because both `groups` and `condition` are
-- reserved words in MySQL 8.
-- =============================================================================

CREATE TABLE IF NOT EXISTS ws_sessions (
  code             VARCHAR(8)   NOT NULL,
  slug             VARCHAR(64)  NULL,      -- human-readable name used in join links
  topic            VARCHAR(64)  NOT NULL,
  label            VARCHAR(160) NOT NULL DEFAULT '',
  mode             VARCHAR(16)  NOT NULL DEFAULT 'group',   -- group | solo
  results_public   TINYINT(1)   NOT NULL DEFAULT 0,
  reveal_step      INT          NOT NULL DEFAULT -1,        -- -1 = nothing revealed
  admin_token_hash CHAR(64)     NOT NULL,
  created_at       DATETIME     NOT NULL,
  closed_at        DATETIME     NULL,
  PRIMARY KEY (code),
  UNIQUE KEY uq_slug (slug),
  KEY idx_topic_open (topic, closed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS ws_groups (
  id           INT          NOT NULL AUTO_INCREMENT,
  session_code VARCHAR(8)   NOT NULL,
  name         VARCHAR(80)  NOT NULL,
  token_hash   CHAR(64)     NOT NULL,
  created_at   DATETIME     NOT NULL,
  updated_at   DATETIME     NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_session_name (session_code, name),
  CONSTRAINT fk_group_session FOREIGN KEY (session_code)
    REFERENCES ws_sessions (code) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Current state: exactly one row per group and lever.
CREATE TABLE IF NOT EXISTS ws_answers (
  id             INT          NOT NULL AUTO_INCREMENT,
  group_id       INT          NOT NULL,
  lever_id       VARCHAR(64)  NOT NULL,
  value          DOUBLE       NOT NULL,
  confidence     TINYINT      NULL,                        -- 1 hunch .. 3 confident
  condition_text VARCHAR(280) NULL,
  updated_at     DATETIME     NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_group_lever (group_id, lever_id),
  CONSTRAINT fk_answer_group FOREIGN KEY (group_id)
    REFERENCES ws_groups (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Append-only trace: one row per move of the slider. This is what lets a reveal
-- say "the group started at 1.4 and settled on 1.7", and what keeps a record of
-- the discussion after the workshop.
CREATE TABLE IF NOT EXISTS ws_answer_log (
  id         BIGINT      NOT NULL AUTO_INCREMENT,
  group_id   INT         NOT NULL,
  lever_id   VARCHAR(64) NOT NULL,
  value      DOUBLE      NOT NULL,
  confidence TINYINT     NULL,
  created_at DATETIME    NOT NULL,
  PRIMARY KEY (id),
  KEY idx_group_lever (group_id, lever_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 评测平台初始 schema：领域模型（profile / batch / run）

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE agent_profiles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL UNIQUE,
    scaffold        TEXT NOT NULL,                 -- mock | mini-claude-code | langgraph | openai-agents
    model           TEXT NOT NULL,
    harness_version TEXT NOT NULL DEFAULT 'dev',
    sandbox_image   TEXT NOT NULL DEFAULT 'local', -- 生产必须是镜像 digest 引用（环境指纹，保证批次可比）
    config          JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE batches (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    profile_id      UUID NOT NULL REFERENCES agent_profiles(id),
    status          TEXT NOT NULL DEFAULT 'running',  -- pending | running | done | cancelled
    parallelism     INT  NOT NULL DEFAULT 4,
    priority        INT  NOT NULL DEFAULT 0,
    cases           JSONB NOT NULL,                   -- [{case_id, task, expectations}]
    idempotency_key TEXT UNIQUE,                      -- 幂等提交：同 key 重复请求返回已有批次
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE runs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id         UUID NOT NULL REFERENCES batches(id),
    case_id          TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'queued',
    -- 生命周期：queued -> leased -> running -> passed|failed|error|timeout
    score            REAL,
    cost_usd         REAL,
    turns            INT,
    input_tokens     BIGINT,
    output_tokens    BIGINT,
    trace_path       TEXT,
    error            TEXT,
    runner_id        TEXT,
    retries          INT NOT NULL DEFAULT 0,
    lease_expires_at TIMESTAMPTZ,
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_runs_batch_status ON runs(batch_id, status);
CREATE INDEX idx_runs_lease ON runs(status, lease_expires_at)
    WHERE status IN ('leased', 'running');

-- 开箱即用的演示 profile
INSERT INTO agent_profiles (name, scaffold, model)
VALUES ('mock-demo', 'mock', 'mock-model-v1');

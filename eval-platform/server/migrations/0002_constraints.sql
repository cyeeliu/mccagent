-- 调度状态、评分与批次形态的生产约束。

ALTER TABLE agent_profiles
    ADD CONSTRAINT agent_profiles_scaffold_check
    CHECK (scaffold IN ('mock', 'anthropic', 'langgraph', 'openai-agents', 'mini-claude-code'));

ALTER TABLE batches
    ADD CONSTRAINT batches_status_check
    CHECK (status IN ('pending', 'running', 'done', 'cancelled')),
    ADD CONSTRAINT batches_parallelism_check
    CHECK (parallelism BETWEEN 1 AND 256),
    ADD CONSTRAINT batches_cases_array_check
    CHECK (jsonb_typeof(cases) = 'array' AND jsonb_array_length(cases) > 0);

ALTER TABLE runs
    ADD CONSTRAINT runs_status_check
    CHECK (status IN ('queued', 'leased', 'running', 'passed', 'failed', 'error', 'timeout')),
    ADD CONSTRAINT runs_score_check
    CHECK (score IS NULL OR (score >= 0 AND score <= 1)),
    ADD CONSTRAINT runs_cost_check
    CHECK (cost_usd IS NULL OR cost_usd >= 0),
    ADD CONSTRAINT runs_turns_check
    CHECK (turns IS NULL OR turns >= 0),
    ADD CONSTRAINT runs_token_counts_check
    CHECK (
        (input_tokens IS NULL OR input_tokens >= 0)
        AND (output_tokens IS NULL OR output_tokens >= 0)
    ),
    ADD CONSTRAINT runs_retries_check
    CHECK (retries >= 0),
    ADD CONSTRAINT runs_batch_case_unique
    UNIQUE (batch_id, case_id);

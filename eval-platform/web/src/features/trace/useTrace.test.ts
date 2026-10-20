import { describe, expect, it } from "vitest";
import type { TraceEvent } from "../../api/schemas";
import { mergeBySeq } from "./useTrace";

function chunk(seq: number, delta: string): TraceEvent {
  return { schema_version: 1, seq, ts: 0, type: "llm_chunk", turn: 0, delta };
}

function unknown(seq: number) {
  return { type: "unknown" as const, seq, raw: { whatever: true } };
}

describe("mergeBySeq", () => {
  it("历史与实时乱序输入时按 seq 排序输出", () => {
    const merged = mergeBySeq([chunk(3, "c"), chunk(1, "a")], [chunk(2, "b")]);
    expect(merged.map((e) => (e.type === "llm_chunk" ? e.delta : ""))).toEqual(["a", "b", "c"]);
  });

  it("两段重叠（SSE 与 REST 竞态）时按 seq 去重", () => {
    const merged = mergeBySeq([chunk(1, "a"), chunk(2, "b")], [chunk(2, "b"), chunk(3, "c")]);
    expect(merged).toHaveLength(3);
    expect(merged.map((e) => e.seq)).toEqual([1, 2, 3]);
  });

  it("未知事件（seq<0）不去重，避免后端新事件被吞", () => {
    const merged = mergeBySeq([], [unknown(-1), unknown(-1)]);
    expect(merged).toHaveLength(2);
  });

  it("空输入返回空数组", () => {
    expect(mergeBySeq([], [])).toEqual([]);
  });
});

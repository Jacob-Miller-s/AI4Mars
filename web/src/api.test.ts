import { afterEach, describe, expect, it, vi } from "vitest";
import { getSamples, runStreamUrl } from "./api";

describe("research console API client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("serializes paged split filters for prediction samples", async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ available: true, total: 0, offset: 4, limit: 4, available_splits: [], samples: [] })
    }));
    vi.stubGlobal("fetch", fetchMock);

    await getSamples("run with spaces", { bigRockFalseNegative: true, bigRockToSoil: false, sortBy: "loss", split: "validation" }, 4, 4);

    const requestUrl = String(fetchMock.mock.calls[0][0]);
    expect(requestUrl).toContain("/api/runs/run%20with%20spaces/samples?");
    expect(requestUrl).toContain("offset=4");
    expect(requestUrl).toContain("limit=4");
    expect(requestUrl).toContain("sort_by=loss");
    expect(requestUrl).toContain("big_rock_false_negative=true");
    expect(requestUrl).toContain("split=validation");
  });

  it("uses the configured API path for durable event streams", () => {
    expect(runStreamUrl("run with spaces", 7)).toBe("/api/runs/run%20with%20spaces/stream?after=7");
  });
});
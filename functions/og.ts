import { ImageResponse, loadGoogleFont } from "workers-og";
import { initSync, decompress as brotliDecompress } from "brotli-dec-wasm/web";
import brotliWasm from "brotli-dec-wasm/web/bg.wasm";

// ── Duplicated from src/hooks/useModelData.ts — keep in sync ──
const MODEL_PREFIXES: Record<
  string,
  Partial<Record<"input" | "output", string>>
> = {
  "qwen3-30b-a3b": {
    input: "/data/qwen3-30b-a3b-input",
    output: "/data/qwen3-30b-a3b-output",
  },
  "llama-3.1-8b": {
    input: "/data/llama-3.1-8b-input",
    output: "/data/llama-3.1-8b-output",
  },
  "gemma-3-4b": {
    input: "/data/gemma-3-4b-input",
  },
};

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  "qwen3-30b-a3b": "Qwen3 30B A3B",
  "llama-3.1-8b": "Llama 3.1 8B",
  "gemma-3-4b": "Gemma 3 4B",
};

// ── Duplicated from src/util.ts — keep in sync ──
function visualizeToken(s: string): string {
  return s
    .replace(/ /g, "\u00B7")
    .replace(/\n/g, "\u21B5")
    .replace(/\t/g, "\u21E5");
}

// ── Duplicated from src/types.ts — keep in sync ──
interface ShardManifest {
  vocabSize: number;
  shardSize: number;
  numShards: number;
  k: number;
}
type KnnShard = [number, number][][];

interface Env {
  ASSETS: Fetcher;
}

// Minimal React-element builder for satori (avoids workers-og HTML parser bugs)
type SatoriNode = string | SatoriElement;
interface SatoriElement {
  type: string;
  props: {
    style?: Record<string, unknown>;
    children?: SatoriNode | SatoriNode[];
    [key: string]: unknown;
  };
}
function h(
  type: string,
  style: Record<string, unknown>,
  ...children: SatoriNode[]
): SatoriElement {
  return {
    type,
    props: {
      style: { display: "flex", ...style },
      children:
        children.length === 0
          ? undefined
          : children.length === 1
            ? children[0]
            : children,
    },
  };
}

/** Load a font file from static assets. */
async function loadFont(
  env: Env,
  baseUrl: string,
  path: string
): Promise<ArrayBuffer> {
  const url = new URL(path, baseUrl);
  const resp = await env.ASSETS.fetch(url.toString());
  if (!resp.ok) throw new Error(`Failed to load font: ${path}`);
  return resp.arrayBuffer();
}

/**
 * Fetch a static asset and parse as JSON, decompressing brotli if needed.
 * Uses first-byte detection (same approach as src/hooks/useModelData.ts).
 */
async function fetchAsset<T>(
  env: Env,
  baseUrl: string,
  path: string
): Promise<T> {
  const url = new URL(path, baseUrl);
  const resp = await env.ASSETS.fetch(url.toString());
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${path}`);

  const bytes = new Uint8Array(await resp.arrayBuffer());
  const firstByte = bytes[0];

  let json: string;
  if (firstByte === 0x7b || firstByte === 0x5b) {
    json = new TextDecoder().decode(bytes);
  } else {
    initSync(brotliWasm); // no-op after first call
    json = new TextDecoder().decode(brotliDecompress(bytes));
  }

  return JSON.parse(json);
}

// ── Style constants (matching App.css design system) ──
const display = "'Syne'";
const mono = "'JetBrains Mono'";
const body = "'DM Sans'";
const bgDeep = "#080b10";
const textPrimary = "#e8edf4";
const textSecondary = "#7c8898";
const textMuted = "#4f5b6b";
const accentToken = "#f0b866";
const accentSimilarity = "#8ea4cc";
const accentTokenDim = "rgba(240, 184, 102, 0.08)";
const accentSimDim = "rgba(142, 164, 204, 0.4)";
const border = "#1c2636";

function buildNeighborRow(
  rank: number,
  nStr: string,
  nId: number,
  simStr: string,
  barWidth: number
): SatoriElement {
  return h(
    "div",
    { display: "flex", alignItems: "center", padding: "21px 20px" },
    // Rank
    h(
      "div",
      { width: 40, color: textMuted, fontFamily: mono, fontSize: 29 },
      `${rank}`
    ),
    // Token name
    h(
      "div",
      {
        flex: 1,
        color: accentToken,
        fontFamily: mono,
        fontSize: 32,
        overflow: "hidden",
        whiteSpace: "nowrap",
        textOverflow: "ellipsis",
      },
      nStr
    ),
    // Token ID (separate column)
    h(
      "div",
      {
        width: 120,
        color: textMuted,
        fontFamily: mono,
        fontSize: 22,
        marginLeft: 8,
      },
      `#${nId}`
    ),
    // Similarity bar + value
    h(
      "div",
      {
        display: "flex",
        alignItems: "center",
        width: 260,
        justifyContent: "flex-end",
      },
      h(
        "div",
        {
          width: 130,
          height: 8,
          background: border,
          borderRadius: 4,
          marginRight: 10,
          overflow: "hidden",
          display: "flex",
        },
        h("div", {
          width: `${barWidth}%`,
          height: "100%",
          background: accentSimDim,
          borderRadius: 4,
        })
      ),
      h(
        "div",
        {
          color: accentSimilarity,
          fontFamily: mono,
          fontSize: 27,
          width: 110,
          textAlign: "right",
        },
        simStr
      )
    )
  );
}

export const onRequestGet: PagesFunction<Env> = async (context) => {
  const url = new URL(context.request.url);
  const model = url.searchParams.get("model");
  const embedding = (url.searchParams.get("embedding") || "input") as
    | "input"
    | "output";
  const tokenParam = url.searchParams.get("token");

  if (!model || tokenParam === null) {
    return new Response("Missing required params: model, token", {
      status: 400,
    });
  }

  const tokenId = parseInt(tokenParam, 10);
  if (isNaN(tokenId) || tokenId < 0) {
    return new Response("Invalid token ID", { status: 400 });
  }

  const prefix = MODEL_PREFIXES[model]?.[embedding];
  if (!prefix) {
    return new Response(`Unknown model/embedding: ${model}/${embedding}`, {
      status: 404,
    });
  }

  // Check cache
  const cache = caches.default;
  const cacheKey = new Request(url.toString());
  const cached = await cache.match(cacheKey);
  if (cached) return cached;

  // Load manifest + tokens in parallel, then the specific shard
  const baseUrl = url.origin;
  const [manifest, tokens] = await Promise.all([
    fetchAsset<ShardManifest>(context.env, baseUrl, `${prefix}-manifest.json`),
    fetchAsset<string[]>(context.env, baseUrl, `${prefix}-tokens.json.br`),
  ]);

  if (tokenId >= manifest.vocabSize) {
    return new Response(
      `Token ID ${tokenId} out of range (vocab size: ${manifest.vocabSize})`,
      { status: 404 }
    );
  }

  const shardIndex = Math.floor(tokenId / manifest.shardSize);
  const shard = await fetchAsset<KnnShard>(
    context.env,
    baseUrl,
    `${prefix}-knn-${shardIndex}.json.br`
  );

  const tokenStr = tokens[tokenId];
  const localIndex = tokenId - shardIndex * manifest.shardSize;
  const neighbors = shard[localIndex].slice(0, 5);
  const displayModel = MODEL_DISPLAY_NAMES[model] || model;
  const vizToken = visualizeToken(tokenStr);

  // Load fonts from static assets (bundled in public/fonts/)
  const [syne, jetbrainsMono, dmSans] = await Promise.all([
    loadFont(context.env, baseUrl, "/fonts/Syne-Bold.ttf"),
    loadFont(context.env, baseUrl, "/fonts/JetBrainsMono-Regular.ttf"),
    loadFont(context.env, baseUrl, "/fonts/DMSans-Medium.ttf"),
  ]);

  // Detect non-Latin scripts in token text and load appropriate fallback fonts.
  // Each loadGoogleFont call uses the `text` param so only needed glyphs are fetched.
  const allTokenText = [
    vizToken,
    ...neighbors.map(([nId]) => visualizeToken(tokens[nId])),
  ].join("");

  const scriptFonts: { family: string; regex: RegExp }[] = [
    { family: "Noto Sans SC", regex: /[\u4E00-\u9FFF\u3400-\u4DBF]/ },
    { family: "Noto Sans JP", regex: /[\u3040-\u309F\u30A0-\u30FF]/ },
    { family: "Noto Sans KR", regex: /[\uAC00-\uD7AF]/ },
    { family: "Noto Sans Arabic", regex: /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/ },
    { family: "Noto Sans Devanagari", regex: /[\u0900-\u097F]/ },
    { family: "Noto Sans Thai", regex: /[\u0E00-\u0E7F]/ },
    { family: "Noto Sans Hebrew", regex: /[\u0590-\u05FF]/ },
  ];

  const neededFonts = scriptFonts.filter((sf) => sf.regex.test(allTokenText));
  const fallbackFonts: { name: string; data: ArrayBuffer }[] = [];
  await Promise.all(
    neededFonts.map(async (sf) => {
      try {
        const data = await loadGoogleFont({
          family: sf.family,
          weight: 400,
          text: allTokenText,
        });
        fallbackFonts.push({ name: sf.family, data });
      } catch {
        // Unavailable (e.g., local dev) — missing glyphs show as boxes
      }
    })
  );

  // Compute similarity bar normalization (30%–100% range)
  const maxSim = neighbors[0]?.[1] ?? 1;
  const minSim = neighbors[neighbors.length - 1]?.[1] ?? 0;
  const simRange = maxSim - minSim || 1;

  // Build element tree (React-like objects for satori — avoids workers-og
  // HTML parser bugs that create spurious text nodes breaking flex layout)
  const neighborElements = neighbors.map(([nId, sim], i) => {
    const nStr = visualizeToken(tokens[nId]);
    const simStr = sim.toFixed(4);
    const barWidth = 30 + ((sim - minSim) / simRange) * 70;
    return buildNeighborRow(i + 1, nStr, nId, simStr, barWidth);
  });

  const element = h(
    "div",
    {
      display: "flex",
      flexDirection: "column",
      width: 1200,
      height: 630,
      background: bgDeep,
      padding: "32px 40px",
    },
    // Header
    h(
      "div",
      { display: "flex", flexDirection: "column", marginBottom: 18 },
      h(
        "div",
        {
          fontSize: 34,
          fontWeight: 700,
          color: textPrimary,
          fontFamily: display,
        },
        "Token Embeddings KNN Explorer"
      ),
      h(
        "div",
        {
          fontSize: 22,
          color: textSecondary,
          fontFamily: body,
          marginTop: 4,
        },
        `${displayModel} \u00B7 ${embedding} embeddings`
      )
    ),
    // Selected token
    h(
      "div",
      {
        display: "flex",
        alignItems: "center",
        background: accentTokenDim,
        borderLeft: `3px solid ${accentToken}`,
        padding: "14px 18px",
        borderRadius: "0 7px 7px 0",
        marginBottom: 14,
      },
      h(
        "div",
        {
          fontSize: 17,
          color: textSecondary,
          fontFamily: body,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          marginRight: 10,
        },
        `Token #${tokenId}`
      ),
      h(
        "div",
        { fontSize: 32, color: accentToken, fontFamily: mono },
        vizToken
      )
    ),
    // Neighbors
    h(
      "div",
      { display: "flex", flexDirection: "column", flex: 1 },
      h(
        "div",
        {
          fontSize: 16,
          color: textMuted,
          fontFamily: body,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          padding: "0 18px",
          marginBottom: 5,
        },
        "Nearest Neighbors"
      ),
      ...neighborElements
    )
  );

  const imageResponse = new ImageResponse(
    element as unknown as string,
    {
      width: 1200,
      height: 630,
      fonts: [
        {
          name: "Syne",
          data: syne,
          weight: 700,
          style: "normal" as const,
        },
        {
          name: "JetBrains Mono",
          data: jetbrainsMono,
          weight: 400,
          style: "normal" as const,
        },
        {
          name: "DM Sans",
          data: dmSans,
          weight: 500,
          style: "normal" as const,
        },
        ...fallbackFonts.map((f) => ({
          name: f.name,
          data: f.data,
          weight: 400 as const,
          style: "normal" as const,
        })),
      ],
      emoji: "twemoji",
    }
  );

  const imageBuffer = await imageResponse.arrayBuffer();
  const response = new Response(imageBuffer, {
    headers: {
      "Content-Type": "image/png",
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });

  context.waitUntil(cache.put(cacheKey, response.clone()));
  return response;
};

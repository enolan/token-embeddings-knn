// ── Duplicated from functions/og.ts — keep in sync ──
const MODEL_DISPLAY_NAMES: Record<string, string> = {
  "qwen3-30b-a3b": "Qwen3 30B A3B",
  "llama-3.1-8b": "Llama 3.1 8B",
  "gemma-3-4b": "Gemma 3 4B",
};

/**
 * Middleware that rewrites OG meta tags when the URL contains ?model= and ?token= params.
 * No data loading — just string manipulation for fast response.
 */
export const onRequest: PagesFunction = async (context) => {
  const response = await context.next();

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("text/html")) return response;

  const url = new URL(context.request.url);
  const model = url.searchParams.get("model");
  const token = url.searchParams.get("token");
  if (!model || !token) return response;

  const embedding = url.searchParams.get("embedding") || "input";
  const displayModel = MODEL_DISPLAY_NAMES[model] || model;
  const ogImageUrl = `${url.origin}/og?model=${encodeURIComponent(model)}&embedding=${encodeURIComponent(embedding)}&token=${encodeURIComponent(token)}`;
  const canonicalUrl = url.toString();
  const title = `Token #${token} \u2014 ${displayModel} \u2014 Token Embeddings KNN`;
  const description = `Nearest neighbors of token #${token} in ${displayModel} ${embedding} embeddings by cosine similarity`;

  return new HTMLRewriter()
    .on("title", {
      element(element) {
        element.setInnerContent(title);
      },
    })
    .on('meta[property="og:url"]', {
      element(element) {
        element.setAttribute("content", canonicalUrl);
      },
    })
    .on('meta[property="og:title"]', {
      element(element) {
        element.setAttribute("content", title);
      },
    })
    .on('meta[property="og:description"]', {
      element(element) {
        element.setAttribute("content", description);
      },
    })
    .on('meta[property="og:image"]', {
      element(element) {
        element.setAttribute("content", ogImageUrl);
      },
    })
    .on('meta[property="og:image:width"]', {
      element(element) {
        element.setAttribute("content", "1200");
      },
    })
    .on('meta[property="og:image:height"]', {
      element(element) {
        element.setAttribute("content", "630");
      },
    })
    .on('meta[name="twitter:image"]', {
      element(element) {
        element.setAttribute("content", ogImageUrl);
      },
    })
    .on('meta[name="description"]', {
      element(element) {
        element.setAttribute("content", description);
      },
    })
    .transform(response);
};

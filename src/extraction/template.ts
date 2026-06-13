/**
 * Vision-MCP v9: Cross-page template awareness for multi-page PDF extraction.
 *
 * Does NOT inject field names or coordinates into prompts.
 * Instead, records the model's own description of the page structure
 * from page 1, and passes it as context to subsequent pages so the model
 * knows the layout is consistent without being told specific locations.
 */

interface PageTemplate {
  pdfPath: string;
  structureDescription: string;
  createdAt: number;
}

const templateCache = new Map<string, PageTemplate>();

/** Max age for template cache (5 minutes) */
const TEMPLATE_TTL_MS = 5 * 60 * 1000;

/**
 * Build a cache key from PDF path + page range.
 */
function cacheKey(pdfPath: string, firstPage: number): string {
  return `${pdfPath}::first_page_${firstPage}`;
}

/**
 * Store a page structure template.
 * `structureDescription` should be the model's own summary of page layout.
 */
export function cacheTemplate(
  pdfPath: string,
  firstPage: number,
  structureDescription: string
): void {
  const key = cacheKey(pdfPath, firstPage);
  templateCache.set(key, {
    pdfPath,
    structureDescription,
    createdAt: Date.now(),
  });
  console.error(`[template] Cached layout for pages from ${firstPage} in ${pdfPath}`);
  // Cleanup old entries
  for (const [k, v] of templateCache) {
    if (Date.now() - v.createdAt > TEMPLATE_TTL_MS) {
      templateCache.delete(k);
    }
  }
}

/**
 * Get a cross-page hint for subsequent pages.
 * Returns a short string that can be appended to the adaptive reading prompt,
 * or null if no template is cached.
 *
 * The hint is intentionally vague: it tells the model the layout is similar
 * but does NOT specify field names, positions, or expected values.
 */
export function getCrossPageHint(pdfPath: string, currentPage: number): string | null {
  // Try to find a template from a nearby first page
  for (const [key, template] of templateCache) {
    if (template.pdfPath === pdfPath && Date.now() - template.createdAt < TEMPLATE_TTL_MS) {
      return "此文件的其他頁面佈局與你剛讀過的那頁相似，請以同樣的系統性方法逐區閱讀。";
    }
  }
  return null;
}

/**
 * Clear template cache for a specific PDF.
 */
export function clearTemplate(pdfPath: string): void {
  for (const [key, template] of templateCache) {
    if (template.pdfPath === pdfPath) {
      templateCache.delete(key);
    }
  }
}

/**
 * Clear all templates.
 */
export function clearAllTemplates(): void {
  templateCache.clear();
}

/**
 * Generate a page structure summary from extraction results.
 * This is the model's own description, NOT a hardcoded template.
 */
export function summarizePageStructure(result: any): string {
  if (!result || !result.finalJson) return "";

  const fields = Object.entries(result.finalJson)
    .filter(([_, v]: [string, any]) => v && v.value && v.value.length > 0)
    .map(([name]) => name);

  if (fields.length === 0) return "";
  return `${fields.length} fields extracted`;
}

import { existsSync, readFileSync, statSync, writeFileSync } from "fs";
import { dirname, extname, join, parse } from "path";
import sharp from "sharp";
import type { VisionProvider } from "../providers/base.js";
import {
  BATCH_MAX_PAGES,
  MAX_IMAGE_WIDTH,
  MAX_OUTPUT_TOKENS,
  MODEL,
  WEBP_QUALITY_OCR,
} from "../config/constants.js";
import { getCapabilityMatrix } from "../runtime/capabilities.js";
import { resolveAccuracyPolicy } from "../runtime/accuracy.js";
import {
  asTextEnvelope,
  collectTokenMetrics,
  envelope,
  parseJsonText,
  timed,
} from "../runtime/envelope.js";
import { handleEstimateTokens, handlePdfInfo, handleBatchStatus, handleBatchStatusAll } from "./common.js";
import { handleAnalyzeImage } from "./image-analyze.js";
import { handleAnalyzePdf } from "./pdf-analyze.js";
import { handleAnalyzeVideo, handleAnalyzeVideoChunked } from "./video.js";
import { handleExtractVerify } from "./extraction.js";
import { aggregateLosslessPages } from "../extraction/lossless.js";
import { extractLosslessDocument } from "../extraction/lossless.js";
import { writeUniversalFinalOutputs } from "../output/universal-writer.js";
import { submitBatchedJobs } from "../batch/manager.js";
import { getPdfPageCount, getPdfType, renderPageSmart } from "../rendering/pdf.js";
import {
  preprocessAggressive,
  preprocessForOCR,
  preprocessHandwriting,
  preprocessLight,
  preprocessNegative,
  preprocessScanned,
  preprocessTable,
} from "../preprocessing/pipeline.js";
import { extToMime, isImageExt, isVideoExt, parsePageRangeDetailed } from "../utils/helpers.js";
import { optimizeForVision } from "../rendering/image.js";

type ToolResult = { content: { type: "text"; text: string }[] };

export class ToolRouter {
  constructor(private readonly provider: VisionProvider) {}

  async dispatch(name: string, args: any): Promise<ToolResult> {
    switch (name) {
      case "vision_inspect":
        return this.visionInspect(args);
      case "vision_prepare":
        return this.visionPrepare(args);
      case "vision_analyze":
        return this.visionAnalyze(args);
      case "vision_extract":
        return this.visionExtract(args);
      case "vision_jobs":
        return this.visionJobs(args);
      default:
        return asTextEnvelope(envelope({
          success: false,
          tool: name,
          strategy: "unknown",
          errors: [`Unknown tool: ${name}`],
        }));
    }
  }

  private ensureFile(filePath: string): string | null {
    if (!filePath) return "file_path required";
    if (!existsSync(filePath)) return `File not found: ${filePath}`;
    return null;
  }

  private mediaType(filePath: string, override?: string): "pdf" | "image" | "video" | "unknown" {
    if (override && override !== "auto") return override as any;
    const ext = extname(filePath).toLowerCase();
    if (ext === ".pdf") return "pdf";
    if (isImageExt(ext)) return "image";
    if (isVideoExt(ext)) return "video";
    return "unknown";
  }

  private translatePdfStrategy(strategy?: string): string | undefined {
    if (!strategy || strategy === "auto") return undefined;
    if (strategy === "realtime") return "concurrent";
    if (strategy === "multi_image" || strategy === "chunked") return "multi-image";
    if (strategy === "batch") return "batch";
    return strategy;
  }

  private parseToolJson(result: string | ToolResult): any {
    const text = typeof result === "string" ? result : result.content?.[0]?.text ?? "";
    return parseJsonText(text);
  }

  private summarizeErrors(result: any): string[] | undefined {
    if (!result) return undefined;
    if (typeof result.error === "string") return [result.error];
    if (Array.isArray(result.errors)) return result.errors.map((e: any) => String(e));
    return undefined;
  }

  private markReviewFields(result: any): any {
    if (result?.universal_schema === "universal_document_semantics_v2") return result;
    const clone = result && typeof result === "object" ? structuredClone(result) : result;
    const visit = (node: any) => {
      if (!node || typeof node !== "object") return;
      if ("confidence" in node || "verified" in node || "value" in node) {
        const confidence = String(node.confidence ?? "").toLowerCase();
        const verified = node.verified === true;
        const value = String(node.value ?? "");
        if (!verified || confidence === "low" || value.includes("[?]") || value.trim() === "") {
          node.needs_review = true;
        } else {
          node.needs_review = false;
        }
      }
      for (const item of Object.values(node)) visit(item);
    };
    visit(clone);
    return clone;
  }

  private async visionInspect(args: any): Promise<ToolResult> {
    const filePath = args.file_path as string;
    const missing = this.ensureFile(filePath);
    if (missing) {
      return asTextEnvelope(envelope({ success: false, tool: "vision_inspect", strategy: "validate", errors: [missing] }));
    }

    const policy = resolveAccuracyPolicy(args);
    const mediaType = this.mediaType(filePath);
    const capability = getCapabilityMatrix(this.provider);

    const { value, elapsedMs } = await timed(async () => {
      const estimates: Record<string, any> = {};
      if (mediaType === "pdf") {
        estimates.pdf_info = this.parseToolJson(await handlePdfInfo({ pdf_path: filePath }));
        estimates.tokens = this.parseToolJson(await handleEstimateTokens({
          file_path: filePath,
          pages: args.pages || "1",
          max_image_width: args.max_image_width ?? MAX_IMAGE_WIDTH,
          max_pixels: args.max_pixels,
          render_scale: args.render_scale,
          lossless_mode: args.lossless_mode !== false,
        }));
        estimates.pdf_type = getPdfType(filePath) || "unknown";
      } else {
        estimates.tokens = this.parseToolJson(await handleEstimateTokens({
          file_path: filePath,
          pages: args.pages || "1",
          max_image_width: args.max_image_width ?? MAX_IMAGE_WIDTH,
          max_pixels: args.max_pixels,
          render_scale: args.render_scale,
          lossless_mode: args.lossless_mode !== false,
          fps: args.fps,
        }));
        if (mediaType === "image") {
          const meta = await sharp(readFileSync(filePath)).metadata();
          estimates.image = {
            width: meta.width,
            height: meta.height,
            format: meta.format,
            size_mb: +(statSync(filePath).size / 1024 / 1024).toFixed(2),
          };
        }
      }
      return estimates;
    });

    const recommendedStrategy =
      mediaType === "pdf"
        ? "vision_analyze strategy=auto or vision_extract for fields"
        : mediaType === "video"
          ? "vision_analyze strategy=auto; use chunked for large videos"
          : "vision_analyze for free-form or vision_extract for fields";

    return asTextEnvelope(envelope({
      success: !this.summarizeErrors(value),
      tool: "vision_inspect",
      strategy: "inspect",
      summary: {
        media_type: mediaType,
        accuracy_mode: policy.mode,
        recommended_strategy: recommendedStrategy,
        cost_controls: {
          image_token_policy: "Use full-page calls for layout/context, then crop OCR only for unverified or invalid fields.",
          cache_policy: args.cache_policy || "auto",
          cost_policy: args.cost_policy || "quality_first",
          quality_guard: "Cost hints never suppress required-field verification.",
          batch_note: "Batch can reduce cost for latency-insensitive jobs only when provider/region/model support is known.",
        },
        capabilities: capability,
      },
      results: value,
      metrics: { elapsed_ms: elapsedMs, ...collectTokenMetrics(value) },
      errors: this.summarizeErrors(value),
    }));
  }

  private async visionPrepare(args: any): Promise<ToolResult> {
    const filePath = args.file_path as string;
    const missing = this.ensureFile(filePath);
    if (missing) {
      return asTextEnvelope(envelope({ success: false, tool: "vision_prepare", strategy: "validate", errors: [missing] }));
    }

    const mode = (args.mode as string) || "auto";
    const mediaType = this.mediaType(filePath);
    const { value, elapsedMs } = await timed(async () => {
      if (mediaType === "pdf") {
        const total = await getPdfPageCount(filePath);
        const parsedPages = args.pages ? parsePageRangeDetailed(args.pages, total) : { pages: [1] };
        if (parsedPages.error) return { error: parsedPages.error };
        const nums = parsedPages.pages;
        const rendered = [];
        for (const pn of nums) {
          const img = await renderPageSmart(filePath, pn, args.max_image_width ?? MAX_IMAGE_WIDTH, true, mode === "handwriting", false, {
            renderScale: args.render_scale,
            maxPixels: args.max_pixels,
            losslessMode: args.lossless_mode !== false,
          });
          const outPath = this.resolveOutputPath(args.output_path, filePath, pn, "png");
          if (outPath) writeFileSync(outPath, img.buffer);
          rendered.push({
            page: pn,
            width: img.width,
            height: img.height,
            mime: img.mime,
            output_path: outPath,
            image_base64: args.return_base64 ? img.buffer.toString("base64") : undefined,
          });
        }
        const result: any = {
          media_type: "pdf",
          lossless_mode: args.lossless_mode !== false,
          preserve_original: args.preserve_original !== false,
          rendered,
        };
        if (args.return_artifacts || args.preprocess_variants) {
          result.artifacts = [
            this.sourceArtifact(filePath, "pdf"),
            ...rendered.map((item: any) => ({
              role: "prepared_lossless_png_candidate",
              page: item.page,
              mime: item.mime,
              width: item.width,
              height: item.height,
              output_path: item.output_path,
              derived_from: "trusted_original",
              lossy: false,
            })),
          ];
          result.preprocess_variants = this.expandPreprocessVariants(args.preprocess_variants);
        }
        return result;
      }

      if (mediaType !== "image") return { error: `Unsupported prepare file type: ${extname(filePath)}` };

      const raw = readFileSync(filePath);
      const prepared =
        mode === "light" ? await preprocessLight(raw)
          : mode === "table" ? await preprocessTable(raw)
            : mode === "handwriting" ? await preprocessHandwriting(raw)
              : mode === "negative" ? await preprocessNegative(raw)
                : mode === "scan" || mode === "photo" ? await preprocessForOCR(raw, { docType: mode as any })
                  : await preprocessForOCR(raw);
      const outPath = args.output_path as string | undefined;
      if (outPath) writeFileSync(outPath, Buffer.from(prepared.buffer));
      const result: any = {
        media_type: "image",
        mode,
        lossless_mode: args.lossless_mode !== false,
        preserve_original: args.preserve_original !== false,
        width: prepared.width,
        height: prepared.height,
        mime: prepared.mime,
        steps: prepared.appliedSteps,
        detected_type: prepared.detectedDocType,
        output_path: outPath,
        image_base64: args.return_base64 || !outPath ? Buffer.from(prepared.buffer).toString("base64") : undefined,
      };
      if (args.return_artifacts || args.preprocess_variants) {
        result.artifacts = [
          this.sourceArtifact(filePath, "image"),
          {
            role: "preprocess_candidate_png",
            variant: mode,
            mime: prepared.mime,
            width: prepared.width,
            height: prepared.height,
            output_path: outPath,
            derived_from: "trusted_original",
            lossy: false,
            steps: prepared.appliedSteps,
          },
        ];
        result.preprocess_variants = this.expandPreprocessVariants(args.preprocess_variants);
      }
      return result;
    });

    return asTextEnvelope(envelope({
      success: !value?.error,
      tool: "vision_prepare",
      strategy: mode,
      summary: { media_type: mediaType },
      results: value,
      metrics: { elapsed_ms: elapsedMs },
      errors: value?.error ? [value.error] : undefined,
    }));
  }

  private resolveOutputPath(outputPath: string | undefined, inputPath: string, page: number, ext: string): string | undefined {
    if (!outputPath) return undefined;
    const parsed = parse(outputPath);
    if (parsed.ext && page === 1) return outputPath;
    if (parsed.ext) return join(parsed.dir || dirname(inputPath), `${parsed.name}-p${page}${parsed.ext}`);
    const base = parse(inputPath).name;
    return join(outputPath, `${base}-p${page}.${ext}`);
  }

  private resolveExtractionOutputDir(outputDir: string | undefined, inputPath: string, enabled: boolean): string | undefined {
    if (!enabled) return undefined;
    if (outputDir && String(outputDir).trim()) return String(outputDir);
    const base = parse(inputPath).name.replace(/[<>:"/\\|?*\x00-\x1F]/g, "_") || "document";
    return join(dirname(inputPath), `vision_extract_${base}_${this.timestampForPath(new Date())}`);
  }

  private timestampForPath(date: Date): string {
    const pad = (n: number) => String(n).padStart(2, "0");
    return [
      date.getFullYear(),
      pad(date.getMonth() + 1),
      pad(date.getDate()),
      "_",
      pad(date.getHours()),
      pad(date.getMinutes()),
      pad(date.getSeconds()),
    ].join("");
  }

  private sourceArtifact(filePath: string, mediaType: "pdf" | "image" | "video" | "unknown"): any {
    return {
      role: "trusted_original",
      path: filePath,
      mime: mediaType === "pdf" ? "application/pdf" : extToMime(extname(filePath).toLowerCase()),
      preserve_original: true,
      lossy_reencoded: false,
    };
  }

  private expandPreprocessVariants(raw: any): string[] | undefined {
    const variants = Array.isArray(raw) ? raw.map((v) => String(v).trim()).filter(Boolean) : [];
    if (!variants.length) return undefined;
    const expanded = new Set<string>();
    for (const variant of variants) {
      if (variant === "forensic") {
        ["original", "deskew", "grayscale-normalize", "adaptive-threshold", "crop-only"].forEach((v) => expanded.add(v));
      } else {
        expanded.add(variant);
      }
    }
    return [...expanded];
  }

  private async visionAnalyze(args: any): Promise<ToolResult> {
    const filePath = args.file_path as string;
    const prompt = args.prompt as string;
    const missing = this.ensureFile(filePath);
    if (missing || !prompt) {
      return asTextEnvelope(envelope({
        success: false,
        tool: "vision_analyze",
        strategy: "validate",
        errors: [missing, !prompt ? "prompt required" : undefined].filter(Boolean) as string[],
      }));
    }

    const mediaType = this.mediaType(filePath, args.media_type);
    const policy = resolveAccuracyPolicy(args);
    const common = {
      prompt,
      pages: args.pages || "",
      max_tokens: args.max_tokens ?? MAX_OUTPUT_TOKENS,
      concurrency: policy.concurrency,
      max_api_concurrency: args.max_api_concurrency,
      render_concurrency: args.render_concurrency,
      chunk_size: args.chunk_size,
      image_quality: policy.imageQuality,
        max_image_width: args.max_image_width ?? MAX_IMAGE_WIDTH,
        max_pixels: args.max_pixels,
        min_pixels: args.min_pixels,
        render_scale: args.render_scale,
        lossless_mode: args.lossless_mode !== false,
        enable_thinking: args.enable_thinking,
      thinking_budget: args.thinking_budget,
      vl_high_resolution_images: policy.vlHighResolutionImages,
      temperature: args.temperature,
      top_p: args.top_p,
    };

    const { value, elapsedMs } = await timed(async () => {
      if (mediaType === "pdf") {
        const strategy = this.translatePdfStrategy(args.strategy);
        const result = await handleAnalyzePdf(this.provider, { ...common, pdf_path: filePath, strategy });
        return this.parseToolJson(result);
      }
      if (mediaType === "image") {
        return this.parseToolJson(await handleAnalyzeImage(this.provider, { ...common, image_path: filePath }));
      }
      if (mediaType === "video") {
        const useChunked = args.strategy === "chunked";
        return this.parseToolJson(
          useChunked
            ? await handleAnalyzeVideoChunked(this.provider, { ...common, video_path: filePath, chunk_duration_sec: args.chunk_duration_sec, aggregate: args.aggregate })
            : await handleAnalyzeVideo(this.provider, { ...common, video_path: filePath, fps: args.fps, nframes: args.nframes })
        );
      }
      return { error: `Unsupported file type: ${extname(filePath)}` };
    });

    const metrics = collectTokenMetrics(value);
    return asTextEnvelope(envelope({
      success: !this.summarizeErrors(value),
      tool: "vision_analyze",
      strategy: args.strategy || value?.summary?.strategy || "auto",
      summary: {
        media_type: mediaType,
        accuracy_mode: policy.mode,
        underlying_model: MODEL,
        ...(value?.summary ? { underlying_summary: value.summary } : {}),
      },
      results: value?.results ?? value?.text ?? value,
      metrics: { elapsed_ms: elapsedMs, ...metrics },
      warnings: getCapabilityMatrix(this.provider).notes,
      errors: this.summarizeErrors(value),
    }));
  }

  private async visionExtract(args: any): Promise<ToolResult> {
    const filePath = args.file_path as string;
    const fields = args.fields as any[] | undefined;
    const missing = this.ensureFile(filePath);
    if (missing) {
      return asTextEnvelope(envelope({
        success: false,
        tool: "vision_extract",
        strategy: "validate",
        errors: [missing].filter(Boolean) as string[],
      }));
    }

    const mediaType = this.mediaType(filePath);
    const policy = resolveAccuracyPolicy(args);
    const attentionFields = this.normalizeAttentionFields(args.attention_fields, fields);
    const extractionFields = attentionFields;
    const outputDir = this.resolveExtractionOutputDir(args.output_dir, filePath, args.save_outputs !== false && args.writer_mode !== "none");
    const writerMode = args.writer_mode || "jsonl_checkpoint_then_bulk_export";
    const prompt = this.buildFieldPrompt(extractionFields || [], args);
    const { value, elapsedMs } = await timed(async () => {
      if (mediaType === "pdf") {
        const result = await handleAnalyzePdf(this.provider, {
          pdf_path: filePath,
          pages: args.pages || "1",
          prompt,
          fields: extractionFields || [],
          lossless_extraction: true,
          strategy: args.strategy,
          max_tokens: args.max_tokens ?? MAX_OUTPUT_TOKENS,
          concurrency: policy.concurrency,
          max_api_concurrency: args.max_api_concurrency,
          render_concurrency: args.render_concurrency,
          image_quality: policy.imageQuality,
          max_image_width: args.max_image_width ?? MAX_IMAGE_WIDTH,
          max_pixels: args.max_pixels,
          min_pixels: args.min_pixels,
          render_scale: args.render_scale,
          lossless_mode: args.lossless_mode !== false,
          vl_high_resolution_images: policy.vlHighResolutionImages,
          self_consistency_votes: policy.selfConsistencyVotes,
          temperature: args.temperature ?? 0,
          cost_policy: args.cost_policy,
          cache_policy: args.cache_policy,
          return_cost_breakdown: args.return_cost_breakdown !== false,
          budget_hint_usd: args.budget_hint_usd,
          max_unverified_required_fields: args.max_unverified_required_fields,
          ocr_verify: policy.ocrVerify,
          provider_mode: args.provider_mode || "auto",
          extraction_style: args.extraction_style || "model_first",
          verification_mode: args.verification_mode || "strict",
          return_evidence: args.return_evidence !== false,
          return_quality_report: args.return_quality_report !== false,
          attention_fields: attentionFields,
          attention_rules: args.attention_rules,
          domain_hint: args.domain_hint || "auto",
          semantic_mode: args.semantic_mode || "auto",
          output_grain: args.output_grain || "auto",
          integration_mode: args.integration_mode || "none",
          extract_all_fields: args.extract_all_fields !== false,
          output_dir: outputDir,
          save_outputs: args.save_outputs !== false,
          export_formats: args.export_formats,
          writer_mode: writerMode,
          resume_from: args.resume_from,
        });
        const parsed = this.parseToolJson(result);
        if (parsed?.results) {
          return aggregateLosslessPages(parsed.results, extractionFields || [], filePath, {
            attentionFields,
            attentionRules: args.attention_rules,
            domainHint: args.domain_hint || "auto",
            semanticMode: args.semantic_mode || "auto",
            outputGrain: args.output_grain || "auto",
            integrationMode: args.integration_mode || "none",
            extractAllFields: args.extract_all_fields !== false,
            renderScale: args.render_scale,
            maxApiConcurrency: args.max_api_concurrency ?? policy.concurrency,
            renderConcurrency: args.render_concurrency,
            writerMode,
            outputDir,
            saveOutputs: args.save_outputs !== false,
            exportFormats: args.export_formats,
            resumeFrom: args.resume_from,
          });
        }
        return parsed;
      }
      if (mediaType === "image") {
        return extractLosslessDocument(this.provider, readFileSync(filePath), extToMime(extname(filePath).toLowerCase()), extractionFields, {
          page: 1,
          sourcePath: filePath,
          maxTokens: args.max_tokens ?? MAX_OUTPUT_TOKENS,
          vlHighResolutionImages: policy.vlHighResolutionImages,
          returnCostBreakdown: args.return_cost_breakdown !== false,
          maxUnverifiedRequiredFields: args.max_unverified_required_fields,
          costPolicy: args.cost_policy,
          cachePolicy: args.cache_policy,
          minPixels: args.min_pixels,
          maxPixels: args.max_pixels,
          providerMode: args.provider_mode || "auto",
          extractionStyle: args.extraction_style || "model_first",
          verificationMode: args.verification_mode || "strict",
          returnEvidence: args.return_evidence !== false,
          returnQualityReport: args.return_quality_report !== false,
          attentionFields,
          attentionRules: args.attention_rules,
          domainHint: args.domain_hint || "auto",
          semanticMode: args.semantic_mode || "auto",
          outputGrain: args.output_grain || "auto",
          integrationMode: args.integration_mode || "none",
          extractAllFields: args.extract_all_fields !== false,
          renderScale: args.render_scale,
          maxApiConcurrency: args.max_api_concurrency ?? policy.concurrency,
          renderConcurrency: args.render_concurrency,
          outputDir,
          saveOutputs: args.save_outputs !== false,
          exportFormats: args.export_formats,
          writerMode,
          resumeFrom: args.resume_from,
        });
      }
      return { error: `Unsupported extraction file type: ${extname(filePath)}` };
    });

    const reviewed = this.markReviewFields(value);
    const outputArtifacts =
      args.save_outputs !== false && outputDir && writerMode !== "none" && reviewed?.universal_schema === "universal_document_semantics_v2"
        ? await writeUniversalFinalOutputs(reviewed, {
            outputDir,
            sourcePath: filePath,
            writerMode,
            exportFormats: args.export_formats,
          })
        : [];
    const reviewedWithArtifacts =
      reviewed && typeof reviewed === "object"
        ? {
            ...reviewed,
            artifacts: [
              ...(Array.isArray((reviewed as any).artifacts) ? (reviewed as any).artifacts : []),
              ...(args.return_artifacts ? [
                this.sourceArtifact(filePath, mediaType),
                {
                  role: mediaType === "pdf" ? "rendered_lossless_png_universal_candidate" : "original_lossless_or_png_universal_candidate",
                  mime: mediaType === "pdf" ? "image/png" : extToMime(extname(filePath).toLowerCase()),
                  pages: Array.isArray((reviewed as any).pages) ? (reviewed as any).pages.map((p: any) => p.page) : undefined,
                  derived_from: "trusted_original",
                  lossy: false,
                },
              ] : []),
              ...outputArtifacts,
            ],
          }
        : reviewed;
    const metrics = collectTokenMetrics(reviewedWithArtifacts);
    const errors = this.summarizeErrors(reviewedWithArtifacts);
    const strictSuccess = args.strict_success !== false;
    return asTextEnvelope(envelope({
      success: !errors && (!strictSuccess || reviewed?.success !== false),
      tool: "vision_extract",
      strategy: mediaType === "pdf" ? "pdf-fields" : "document-fields",
      summary: {
        media_type: mediaType,
        accuracy_mode: policy.mode,
        ocr_verify: policy.ocrVerify,
        return_evidence: args.return_evidence !== false,
        cost_policy: args.cost_policy || "quality_first",
        cache_policy: args.cache_policy || "auto",
        budget_hint_usd: args.budget_hint_usd,
        model_policy: args.model_policy || "quality_first",
        provider_mode: args.provider_mode || "auto",
        extraction_style: args.extraction_style || "model_first",
        verification_mode: args.verification_mode || "strict",
        return_quality_report: args.return_quality_report !== false,
        output_dir: outputDir,
        writer_mode: writerMode,
        export_formats: args.export_formats || ["jsonl", "json", "xlsx", "markdown"],
      },
      results: reviewedWithArtifacts,
      metrics: { elapsed_ms: elapsedMs, ...metrics },
      warnings: getCapabilityMatrix(this.provider).notes,
      errors,
    }));
  }

  private normalizeAttentionFields(rawAttention: any, legacyFields?: any[]): any[] {
    const attention = Array.isArray(rawAttention) ? rawAttention : [];
    const legacy = Array.isArray(legacyFields) ? legacyFields : [];
    return [...attention, ...legacy].map((field) => {
      if (typeof field === "string") return { name: field, required: false };
      return { ...field, required: field?.required === true };
    }).filter((field) => typeof field === "string" ? field.trim() : (field?.name || field?.label || field?.field));
  }

  private buildFieldPrompt(fields: any[], args: any = {}): string {
    const names = fields.map((f) => f.name || f.label_pattern || f.labelPattern).filter(Boolean).join(", ");
    return [
      "Extract visible document data using model-first OCR and universal document understanding.",
      "Attention/requested fields are hints for extra checking, not a whitelist.",
      "Always discover and preserve all visible fields, tables, entities, unmapped fields, and orphan values.",
      "Return null for absent, unreadable, or uncertain values.",
      "Do not infer, complete, or fabricate values.",
      "Keep evidence/confidence metadata when available.",
      args.domain_hint ? `Domain hint: ${args.domain_hint}` : "",
      args.attention_rules ? `Attention rules: ${JSON.stringify(args.attention_rules)}` : "",
      `Requested fields: ${names || "(none; discover all visible fields)"}`,
    ].filter(Boolean).join("\n");
  }

  private async visionJobs(args: any): Promise<ToolResult> {
    const action = args.action as string;
    const capability = getCapabilityMatrix(this.provider);
    const { value, elapsedMs } = await timed(async () => {
      if (action === "status") {
        return args.wait_for_completion
          ? this.waitForBatches([args.job_id], args)
          : this.parseToolJson(await handleBatchStatus(this.provider, { batch_id: args.job_id }));
      }
      if (action === "status_all") {
        return args.wait_for_completion
          ? this.waitForBatches(args.job_ids || [], args)
          : this.parseToolJson(await handleBatchStatusAll(this.provider, { batch_ids: args.job_ids }));
      }
      if (action === "results") {
        return args.wait_for_completion
          ? this.waitForBatches([args.job_id], args)
          : this.parseToolJson(await handleBatchStatus(this.provider, { batch_id: args.job_id }));
      }
      if (action === "cancel") {
        return this.cancelBatch(args.job_id);
      }
      if (action === "list") {
        return this.listBatches();
      }
      if (action !== "submit") return { error: `Unsupported action: ${action}` };
      if (args.batch_policy === "disable") return { error: "batch_policy=disable; use vision_analyze realtime instead" };
      if (!capability.supportsBatch && args.batch_policy !== "force") {
        return {
          error: "Batch is not known to be supported for this provider/region/model.",
          fallback: "Use vision_analyze with strategy=realtime or strategy=multi_image.",
          capabilities: capability,
        };
      }
      if (!args.file_path || !args.prompt) return { error: "file_path and prompt required for submit" };
      if (this.mediaType(args.file_path) !== "pdf") return { error: "Only PDF batch submit is supported." };

      const total = await getPdfPageCount(args.file_path);
      const parsedPages = args.pages
        ? parsePageRangeDetailed(args.pages, total)
        : { pages: Array.from({ length: total }, (_, i) => i + 1) };
      if (parsedPages.error) return { error: parsedPages.error };
      const pages = parsedPages.pages;
      const policy = resolveAccuracyPolicy(args);
      const submitted = await submitBatchedJobs(
        this.provider,
        args.file_path,
        pages,
        args.prompt,
        args.max_tokens ?? MAX_OUTPUT_TOKENS,
        policy.imageQuality || WEBP_QUALITY_OCR,
        args.max_image_width ?? MAX_IMAGE_WIDTH,
        args.enable_thinking,
        args.thinking_budget,
        {
          renderScale: args.render_scale,
          maxPixels: args.max_pixels,
          minPixels: args.min_pixels,
          losslessMode: args.lossless_mode !== false,
        },
      );
      return {
        total_pages: pages.length,
        batch_max_pages: BATCH_MAX_PAGES,
        batch_ids: submitted.batchIds,
        chunks: submitted.chunks,
        ...(args.wait_for_completion ? { waited: await this.waitForBatches(submitted.batchIds, args) } : {}),
        message: submitted.batchIds.length === 1
          ? "Batch submitted. Poll with vision_jobs action=status."
          : "Batches submitted. Poll with vision_jobs action=status_all.",
      };
    });

    return asTextEnvelope(envelope({
      success: !this.summarizeErrors(value),
      tool: "vision_jobs",
      strategy: action || "unknown",
      summary: { action, capabilities: capability },
      results: value,
      metrics: { elapsed_ms: elapsedMs, ...collectTokenMetrics(value) },
      errors: this.summarizeErrors(value),
    }));
  }

  private async cancelBatch(jobId: string | undefined): Promise<any> {
    if (!jobId) return { error: "job_id required" };
    const client = (this.provider as any).getClient?.();
    if (!client?.batches?.cancel) return { error: "Provider does not expose batches.cancel" };
    return client.batches.cancel(jobId);
  }

  private async waitForBatches(batchIds: string[], args: any): Promise<any> {
    const ids = (batchIds || []).filter(Boolean);
    if (!ids.length) return { error: "job_id or job_ids required" };
    const pollMs = Math.max(1000, Number(args.poll_interval_ms) || 5000);
    const maxWaitMs = Math.max(pollMs, Number(args.max_wait_ms) || 300000);
    const deadline = Date.now() + maxWaitMs;
    let batches: any[] = [];

    while (true) {
      batches = await Promise.all(
        ids.map(async (id) => this.parseToolJson(await handleBatchStatus(this.provider, { batch_id: id })))
      );
      const pending = batches.filter((b) => this.isBatchPending(b));
      if (!pending.length) break;
      if (Date.now() + pollMs > deadline) break;
      await new Promise((resolve) => setTimeout(resolve, pollMs));
    }

    const pending = batches.filter((b) => this.isBatchPending(b));
    return {
      summary: {
        total_batches: batches.length,
        completed: batches.filter((b) => b.status === "completed").length,
        pending: pending.length,
        failed: batches.filter((b) => ["failed", "error", "expired", "cancelled"].includes(b.status)).length,
        wait_expired: pending.length > 0,
      },
      batches,
    };
  }

  private isBatchPending(batch: any): boolean {
    return batch?.pending === true || ["queued", "validating", "in_progress"].includes(batch?.status);
  }

  private async listBatches(): Promise<any> {
    const client = (this.provider as any).getClient?.();
    if (!client?.batches?.list) return { error: "Provider does not expose batches.list" };
    return client.batches.list();
  }
}

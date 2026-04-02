/**
 * Fallback when no platform-specific `executorchModels.*` file applies.
 * Prefer explicit `.ios` / `.android` / `.web` modules for Metro resolution.
 */
export * from "./executorchModels.xnnpack";

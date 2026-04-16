/**
 * iOS: CoreML-targeted ExecuTorch bundles (see scripts/export_models_ios_coreml.sh).
 */
export const BLAZEFACE_MODEL = require("./blazeface_coreml.pte");
export const AGE_MODEL = require("./age_model_coreml.pte");
export const GENDER_MODEL = require("./gender_model_coreml.pte");
export const NSFW_MODEL = require("./nsfw_model_coreml.pte");

// Text Moderation uses martin-ha/toxic-comment-model (DistilBERT) instead of DeBERTa because
// DeBERTa's disentangled attention drops positional inputs in CoreML's gather layer, causing NaN.
// DistilBERT uses standard scaled dot-product attention which CoreML handles correctly.
// Exported with CPU_ONLY + FLOAT32 for integer embedding compatibility. Re-export: python export_text.py --backend coreml
export const TEXT_MODEL = require("./text_model_coreml.pte");

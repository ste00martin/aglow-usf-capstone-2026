/**
 * iOS: CoreML-targeted ExecuTorch bundles (see scripts/export_models_ios_coreml.sh).
 */
export const BLAZEFACE_MODEL = require("./blazeface_coreml.pte");
export const AGE_MODEL = require("./age_model_coreml.pte");
export const GENDER_MODEL = require("./gender_model_coreml.pte");
export const NSFW_MODEL = require("./nsfw_model_coreml.pte");

// The Text Moderation model (DeBERTa) utilizes complex disentangled attention and relative position encodings. 
// These operations do not have stable CPU fallback implementations in CoreML and will produce NaNs on the iOS Simulator.
// However, because the raw XNNPACK bundle is > 536MB, Metro bundler crashes due to Node's internal string limit.
// We are restoring the CoreML variant to immediately fix the crash. To test text moderation, build on a physical iPhone!
export const TEXT_MODEL = require("./text_model_coreml.pte");

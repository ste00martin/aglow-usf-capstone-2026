/**
 * XNNPACK ExecuTorch bundles (see scripts/export_models_xnnpack.sh).
 * Imported only by non-CoreML platform entry points so Metro does not pull these into iOS bundles.
 */
export const BLAZEFACE_MODEL = require("./blazeface.pte");
export const AGE_MODEL = require("./age_model.pte");
export const GENDER_MODEL = require("./gender_model.pte");
export const NSFW_MODEL = require("./nsfw_model.pte");
export const TEXT_MODEL = require("./text_model.pte");

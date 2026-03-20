const mongoose = require("mongoose");

const detectionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",

    index: true, // fast query ke liye
  },
  fileName: {
    type: String,
    index: true,
  },

  imageName: {
    type: String,
    default: "",
  },

  imagePath: {
    type: String,
    default: "",
  },

  screenshotUrl: {
    type: String,
    default: "",
  },

  videoUrl: {
    type: String,
    index: true,
  },

  textNote: {
    type: String,
    index: true,
  },

  object: {
    type: String,
    index: true,
    default: "unknown",
  },

  vehicle: {
    type: String,
    default: "",
  },

  color: {
    type: String,
    default: "",
  },

  helmet: {
    type: String,
    enum: ["helmet", "no_helmet", "unknown"],
    default: "unknown",
  },

  ocrText: {
    type: String,
    index: true,
    default: "",
  },

  confidence: {
    type: Number,
    default: 0,
  },

  trackingId: {
    type: String,
    index: true,
  },

  bbox: {
    type: [Number],
    default: [],
    set: function (v) {
      // ===== SMART CONVERSION =====
      // Handle empty string, null, undefined
      if (!v || v === "" || v === null || v === undefined) {
        return [];
      }

      // If already array, validate length
      if (Array.isArray(v)) {
        return v.length === 4 ? v : [];
      }

      // If string like "x1,y1,x2,y2", convert to array
      if (typeof v === "string") {
        if (v.trim() === "") return []; // ← YEH ADD KARO
        const nums = v
          .split(",")
          .map((n) => parseFloat(n.trim()))
          .filter((n) => !isNaN(n));
        return nums.length === 4 ? nums : [];
      }
      return [];
    },
    validate: {
      validator: function (v) {
        return Array.isArray(v) && (v.length === 0 || v.length === 4);
      },
      message: "bbox must be empty array [] or 4 numbers",
    },
  },

  timestamp: {
    type: Number,
    default: 0,
  },

  processingTime: {
    type: String,
    default: "",
  },

  createdAt: {
    type: Date,
    default: Date.now,
    index: true,
  },
});

module.exports = mongoose.model("Detection", detectionSchema);

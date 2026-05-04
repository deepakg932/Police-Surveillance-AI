const mongoose = require("mongoose");

const detectionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    index: true,
  },

  jobId: {
    type: String,
    index: true,
  },

  // 🔥 NEW (MOST IMPORTANT)
  status: {
    type: String,
    enum: ["processing", "completed", "failed"],
    default: "processing",
    index: true,
  },

  batchId: {
  type: String,
  index: true,
  default: "",
},

isBatch: {
  type: Boolean,
  default: false,
  index: true,
},

mode: {
  type: String,
  default: "",
},

errorMessage: {
  type: String,
  default: "",
},

  fileName: {
    type: String,
    index: true,
  },
  totalVideos: {
  type: Number,
  default: 0,
},

  imageName: {
    type: String,
    default: "",
  },
isJob:{
  type:Boolean,
  default: false,
  index: true,
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
    if (!v || v === "" || v === null || v === undefined) return [];

    if (Array.isArray(v)) {
      const nums = v.map(Number).filter((n) => !isNaN(n));
      return nums.length === 4 ? nums : [];
    }

    if (typeof v === "string") {
      const nums = v
        .split(",")
        .map((n) => Number(n.trim()))
        .filter((n) => !isNaN(n));
      return nums.length === 4 ? nums : [];
    }

    return [];
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
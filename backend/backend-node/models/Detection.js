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

  fileName: {
    type: String,
    index: true,
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
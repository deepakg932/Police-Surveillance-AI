const mongoose = require("mongoose");

const detectionSchema = new mongoose.Schema({

  fileName: String,

  imageName: String,

  imagePath: String,

  screenshotUrl: String,

  videoUrl: {
    type: String,
    index: true
  },

  textNote: String,

  object: {
    type: String,
    index: true
  },

  vehicle: String,

  color: String,

  helmet: {
    type: String,
    enum: ["helmet", "no_helmet", "unknown"],
    default: "unknown"
  },

  ocrText: {
    type: String,
    index: true
  },

  confidence: Number,

  trackingId: {
    type: String,
    index: true
  },

  bbox: {
    type: [Number],
    validate: v => v.length === 4
  },

  timestamp: Number,

  processingTime: String,

  createdAt: {
    type: Date,
    default: Date.now
  }

});

module.exports = mongoose.model("Detection", detectionSchema);